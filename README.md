# lr_dllm

在 **masked LLaDA** 上实验连续 hidden-state reasoning：reasoning 位置不解码成 token，
在 iteration 之间回填 hidden，最后只对 answer 位置做词表投影。底座冻结，使用 LoRA 训练。

**当前是实现原型。** 已通过小型 LLaDA 的 CPU/CUDA 前向、反向、数值一致性和 adapter 重载检查，
并完成四组预训练 LLaDA-1.5 8B 的 100 步子集训练及两组原版 GSM8K 全量单 epoch 训练，
loss 有限、adapter 重载完全一致。这些共享 GPU 上的运行不构成单进程速度基准。
使用 **原版 GSM8K，不使用 GSM8K-Aug**；已完成的全量运行覆盖全部 7,473 条训练数据。
多 epoch 实验默认每个 epoch 验证一次，按独立验证集的最低 loss 保存最佳 LoRA；官方 test 不参与选择。
仓库不包含模型权重、数据集、训练输出或其他实验仓库的依赖。

## 1. 环境与依赖

推荐 Python **3.11**。已有环境能满足下列版本时可以直接复用，无须新建环境。

| 库 | 已测试版本 | 用途 |
| --- | --- | --- |
| PyTorch | 2.6.0，CUDA 12.4 构建 | 模型、梯度、activation checkpointing |
| Transformers | 4.51.3 | LLaDA 自定义模型与 tokenizer 加载 |
| PEFT | 0.15.2 | LoRA |
| Datasets | 3.6.0 | GSM8K 数据准备 |
| Accelerate | 1.14.0 | 低内存权重加载 |
| Safetensors | 0.7.0 | 权重与 adapter 保存 |
| Hugging Face Hub | 0.36.2 | 下载模型文件 |
| Tokenizers | 0.21.4 | 分词 |
| TensorBoard | 2.21.0 | 训练曲线、验证指标与运行配置 |

不需要 FlashAttention、DeepSpeed、bitsandbytes 或 wandb。版本已固定在 `requirements.txt`；
这不是所有间接依赖的完整锁文件。

```bash
git clone https://github.com/suntea233/lr_dllm.git
cd lr_dllm

# 可选：已有合适环境时跳过这两行
conda create -n lr_dllm python=3.11 -y
conda activate lr_dllm

# GPU 示例：安装与机器驱动兼容的 PyTorch 构建
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__); print('CUDA:', torch.cuda.is_available())"
```

纯 CPU 检查可将 PyTorch 下载地址换成 `https://download.pytorch.org/whl/cpu`。
若环境已有兼容的 PyTorch，可保留当前构建，检查其是否满足依赖版本。

下文通过 `bash run.sh ...` 运行。它使用**当前环境的 `python`**，并自动切换到项目目录。
也可以指定解释器：`PYTHON=/path/to/env/bin/python bash run.sh ...`。
不依赖作者机器上的 conda 路径。

## 2. 先跑小模型检查：不用下载 8B 权重

先只下载 LLaDA 的 config、模型 Python 文件和真实 tokenizer，再生成随机小模型：

```bash
bash run.sh prepare_checkpoint.py --metadata-only
bash run.sh make_tiny_checkpoint.py

CUDA_VISIBLE_DEVICES='' bash run.sh check_training.py
CUDA_VISIBLE_DEVICES='' bash run.sh check_readout.py
```

默认位置为 `checkpoints/llada15` 和 `checkpoints/tiny-random-llada`。
小模型有 2 层、hidden size 32，保留真实 LLaDA 词表和 tokenizer，总参数约 812 万。
它是随机模型，检查结果不能作为推理能力指标。

- `check_training.py`：跨轮梯度、冻结底座、答案区始终 MASK、同位置 CE、padding 忽略、
  checkpoint 梯度一致、BF16 反向，以及 FP32/BF16 adapter 无损重载。
- `check_readout.py`：FP32/BF16 × 1/2/3 轮 × checkpoint 开关，共 12 组新旧路径对照；
  比较答案 logits、loss、LoRA 梯度，并检查中间 LM head 确实没有调用。
- 结果写入 `training_checks.json` 和 `readout_checks.json`。

这些命令可以重复运行；`make_tiny_checkpoint.py` 会拒绝覆盖已有模型。
需要重新生成时，用 `--output checkpoints/another-tiny-model`，后续检查加 `--model` 指向该路径。

## 3. 准备原版 GSM8K 全量数据

```bash
bash run.sh prepare_gsm8k.py
bash run.sh check_data.py \
  --model checkpoints/llada15 --data data/gsm8k_full \
  --splits train test --max-prompt 256 --answer-slots 16 --require-all
```

从 `openai/gsm8k` 的 `main` 配置读取官方划分：

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `data/gsm8k_full/train.jsonl` | 7,473 | 完整微调训练集 |
| `data/gsm8k_full/test.jsonl` | 1,319 | 训练结束后的官方 test 划分评估 |

默认不增加合成数据、不划出验证题，也不按问题去重删除官方训练行。
这个默认导出适用于完整 train 训练、最终只测一次 test 的实验；不能把 test 当作每个 epoch 选 checkpoint 的验证集。
当前训练目标仍是原始答案中 `####` 后的最终数字答案和 EOS；原始解题过程保存在 `solution` 字段中供溯源，
不进入当前模型输入或 loss。切换到全量数据不会引入 CoT 蒸馏。
数据目录中的 `manifest.json` 记录来源、划分、样本数和文件 hash，每条样本保留原始划分及行号。
`check_data.py --require-all` 使用真实 tokenizer 检查全部样本，任何长度超限都会报错。
正式训练也使用 `--strict-lengths`，避免因长度过滤而实际只训练了部分数据。
已用 LLaDA tokenizer 核查：train 的最长 prompt 为 240 tokens、答案含 EOS 最长 12 tokens；
test 分别为 209 和 10。`--max-prompt 256 --answer-slots 16` 保留全部 7,473／1,319 条，无跳过样本。

已有 Datasets 缓存时可以离线准备；需要固定来源时，可为在线准备指定 `--revision <dataset-commit>`：

```bash
bash run.sh prepare_gsm8k.py --offline --cache-dir /path/to/datasets/cache
```

若要根据 val loss 选择 checkpoint，需要与训练数据不重叠的独立验证集。
一种可选划分是从官方 train 固定留出 256 条，其余 7,217 条训练；只在决定使用这个划分后运行：

```bash
bash run.sh prepare_gsm8k.py \
  --output data/gsm8k_dev256 --validation-size 256 --split-seed 42
```

输出 `train.jsonl`、`val.jsonl`、`test.jsonl`，分别为 7,217／256／1,319 条；官方 test 保持不变。
如果要保留全部 7,473 条用于训练，就应另提供合适的独立验证集，不得使用已参与训练的题目选 checkpoint。

也支持沿用 [CoLaR 官方原版 GSM8K 预处理脚本](https://github.com/xiaomi-research/colar/blob/main/data_preprocessing/gsm8k.py)
的 `train[:90%]` / `train[90%:]` 划分：按原始顺序取前 6,726 条训练、后 747 条验证，不先打乱；
官方 test 仍为 1,319 条。这与 CoLaR 论文主实验的 GSM8K-Aug 配置需分开看。

```bash
bash run.sh prepare_gsm8k.py \
  --output data/gsm8k_colar_split --validation-size 747 --validation-policy tail
```

这两种 holdout 是不同的实验配置，应在所有对照组间固定同一份划分。

### 可选：仅用于实现检查的小子集

```bash
bash run.sh experiment.py prepare
bash run.sh check_data.py
```

从 `openai/gsm8k` 的官方 **train** 划分中取数据，按 question 去重，以 seed42 打乱：

| 文件 | 数量 | 说明 |
| --- | ---: | --- |
| `data/train64.jsonl` | 64 | 实现检查／快速过拟合实验 |
| `data/train1000.jsonl` | 1000 | 旧版短训练检查，与 train64 为嵌套关系 |
| `data/val128.jsonl` | 128 | 与上述训练集无重复问题 |

这些子集只用于下面的 CPU/GPU 调试命令，不是全量微调的数据入口。
仅取原始答案中 `####` 后的最终答案；不把 CoT 放入训练输入，不使用官方 test 调参。
`data/manifest.json` 记录来源、seed 和样本数；每条记录保留原始行号。
`check_data.py` 使用真实 tokenizer 检查长度、划分和文件 hash。

已有缓存 Arrow 文件时，可以离线准备：

```bash
bash run.sh experiment.py prepare --arrow /path/to/gsm8k-train.arrow
```

未启用 `--strict-lengths` 时，超出 `--max-prompt` 和 `--answer-slots` 的样本会被跳过并打印数量，不静默截断。
答案长度限制包含 EOS。已检查的上述三个子集均未超限。

## 4. 跑通完整训练入口

CPU 上跑两步，只使用两条训练和两条验证样本：

```bash
CUDA_VISIBLE_DEVICES='' bash run.sh experiment.py train \
  --model checkpoints/tiny-random-llada \
  --data data/train64.jsonl --validation data/val128.jsonl \
  --output runs/tiny-cpu-check \
  --device cpu --dtype float32 --cpu-threads 2 \
  --train-limit 2 --eval-limit 2 --steps 2 --accumulation 2 \
  --verify-reload
```

GPU 上跑相同小模型的 BF16 检查：

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh experiment.py train \
  --model checkpoints/tiny-random-llada \
  --data data/train64.jsonl --validation data/val128.jsonl \
  --output runs/tiny-cuda-check \
  --device cuda:0 --dtype bfloat16 \
  --train-limit 2 --eval-limit 2 --steps 2 --accumulation 2 \
  --min-free-gib 4 --max-memory-gib 2 --verify-reload
```

`--max-memory-gib 2` 是小模型调试使用的 PyTorch allocator 上限，不包含 CUDA context 等外部开销，
**不要把这个值用于 8B 训练**。输出目录必须未存在，重复实验请更换 `--output`。

## 5. 原版 GSM8K 全量 8B LoRA 微调

### 优先复用共享盘或已有缓存中的模型

先检查共享盘和已有模型缓存。只要已有完整权重，就直接引用该目录：

```bash
bash run.sh prepare_checkpoint.py \
  --source /path/to/shared/LLaDA-1.5 \
  --output checkpoints/llada15
```

脚本默认为权重创建符号链接，不复制约 16 GB 的分片；配置、tokenizer 和模型 Python 文件可以保留在本地。
运行时共享源目录必须仍可访问，不能删除或移动链接指向的文件。
只有明确需要搬运独立权重副本时，才显式加 `--copy-weights` 并选一个新输出目录。

若源目录缺少 `modeling_llada.py` / `configuration_llada.py`，脚本会在架构配置匹配后，
从 `GSAI-ML/LLaDA-8B-Instruct` 补入模型代码。离线时可加：
`--code-source /path/to/LLaDA-8B-Instruct`。
完整权重索引和分片存在性会被检查，加载时还会检查参数是否完整匹配。
来源、revision 和代码 hash 写入输出目录的 `provenance.json`。

### 没有可复用权重时再从 Hugging Face 下载

```bash
# 不加 --metadata-only，会下载完整权重（约 16 GB）
bash run.sh prepare_checkpoint.py
```

若已执行 metadata-only，可在同一目录补齐权重。下载文件默认位于 Hugging Face 缓存，
`checkpoints/llada15` 的权重文件链接到缓存；请保留缓存。
固定实验建议为下载指定 `--revision <commit>`，必要时也指定 `--code-revision <commit>`。

### 可选：先用小子集运行一个 optimizer step

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh experiment.py train \
  --model checkpoints/llada15 \
  --data data/train64.jsonl --validation data/val128.jsonl \
  --output runs/llada15-one-step \
  --train-limit 1 --eval-limit 1 --steps 1 --accumulation 1 \
  --rounds 2 --latent-slots 16 --answer-slots 16 \
  --rank 8 --dtype bfloat16 --verify-reload
```

### 完整遍历 train，一个 epoch 的一轮／两轮对照

两组均从同一个 LLaDA-1.5 底座新建 LoRA，不加载旧版子集实验的 adapter。
下面依次运行两组；需要并行时，先实测当前机器的训练显存，再分配进程和显存上限。

```bash
for k in 1 2; do
  CUDA_VISIBLE_DEVICES=0 bash run.sh experiment.py train \
    --model checkpoints/llada15 \
    --data data/gsm8k_full/train.jsonl --validation data/gsm8k_full/test.jsonl \
    --validation-role test \
    --output "runs/gsm8k-full-k${k}-seed42-e1" \
    --epochs 1 --accumulation 4 --rounds "$k" \
    --latent-slots 16 --answer-slots 16 --max-prompt 256 \
    --rank 8 --lr 1e-4 --dtype bfloat16 --seed 42 \
    --strict-lengths --skip-initial-eval --eval-every 0 --verify-reload || break
done
```

`--epochs 1` 对训练行做一次带 seed 的随机排列，无放回地遍历全部 7,473 条。
batch 为 1、`--accumulation 4` 时共有 **1,869 次 optimizer update**：最后一次使用剩余 1 条，
loss 按实际累积条数归一化。`--epochs` 与 `--steps` 互斥；旧版 `--steps` 模式仍按有放回抽样运行，
不能把它的步数直接理解为完整 epoch。

这里沿用 `--validation` 这个参数名指定官方 test 文件，并显式设置 `--validation-role test`。
test 模式只在全部训练结束后评估一次，指标写为 `test_*`，不生成 `best_adapter/` 或据此选择 checkpoint。
当前仍使用下文说明的简化答案 EM，未接入官方 evaluation harness。

### 多 epoch 训练：每个 epoch 验证一次，按 val loss 选最佳 LoRA

下面沿用 CoLaR 的原始顺序 90%/10% 划分：先按第 3 节准备 `data/gsm8k_colar_split`，
每个 epoch 遍历 6,726 条训练题、验证 747 条题；官方 1,319 条 test 不参与选模。
若使用其他独立划分，替换两个数据路径，并在所有对照组间保持一致。

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh experiment.py train \
  --model checkpoints/llada15 \
  --data data/gsm8k_colar_split/train.jsonl --validation data/gsm8k_colar_split/val.jsonl \
  --output runs/gsm8k-k2-seed42-e3 \
  --epochs 3 --accumulation 4 --rounds 2 \
  --latent-slots 16 --answer-slots 16 --max-prompt 256 \
  --rank 8 --lr 1e-4 --dtype bfloat16 --seed 42 \
  --strict-lengths --verify-reload
```

epoch 模式默认跳过训练前验证，只在每个 epoch 结束时遍历一次完整验证集；不按训练 step 密集验证。
`adapter/` 保留最近完成的 epoch，`best_adapter/` 仅在 `val_ce` 严格降低时更新。
`best_checkpoint.json` 记录选择指标、定义、值、epoch、step、验证数据路径和逐题预测文件。
这里的 `val_ce` 是**每题答案 token CE 的平均值，再对所有题取平均**，包含 EOS，忽略 `-100`；
同时记录按有效 token 数加权的 CE，但它不参与 best checkpoint 选择。
默认 TensorBoard 开启，无需额外参数；明确不需要事件日志时可加 `--no-tensorboard`。

默认空闲显存门槛是 `--min-free-gib 40`，这是保守的启动保护，**不是实测最低需求**，
也不会预留显存。8B BF16 底座本身约 15 GiB，LoRA 仍需要底座和跨轮激活。
当前没有量化、CPU offload 或多卡分片支持；不要把多张卡的空闲显存简单相加。
已完成的 8B 短训练使用共享 GPU 上的并行进程，其耗时不能视作单进程推理或训练性能基准。
全量数据的最长输入也可能改变显存需求，启动前应按实际长度做一次前向和反向校准。

## 6. 核心流程与配置含义

```text
问题 embedding + 16 个 latent MASK embedding + 16 个 answer MASK embedding
                           ↓ Transformer 第 1 轮
问题 embedding + 缩放后的 latent hidden       + 16 个 answer MASK embedding
                           ↓ Transformer 第 2 轮
                      answer hidden → LM head → 答案 logits
                           ↓ 同位置 CE + EOS
                    梯度穿过两轮，只更新 LoRA
```

两轮意味着 **两次 Transformer forward、一次 latent 回填**，不是每轮再额外执行两次。
答案标签不进入模型输入；所有 answer 位置在各轮输入中都为 MASK。
latent 不经过词表 softmax、不采样 token、不 detach。
回填前按 MASK embedding 的 RMS 幅度缩放，这是待验证的实验选择。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--rounds` | 2 | 完整双向 Transformer 迭代次数 |
| `--latent-slots` | 16 | 持续传播 hidden 的位置数 |
| `--answer-slots` | 16 | 最终答案空间，包含 EOS |
| `--max-prompt` | 256 | 输入问题的 token 上限，不是 generation length |
| `--rank` | 8 | LoRA rank；alpha 为 2 × rank |
| `--accumulation` | 4 | batch1 下累积几次梯度再更新 |
| `--epochs` | 未指定 | 无放回遍历全部训练数据的轮数，与 `--steps` 互斥 |
| `--steps` | 100（未指定 epochs 时） | 旧版有放回抽样的 optimizer update 数 |
| `--readout` | `final_only` | 中间跳过 LM head，最后只投影 answer |
| `--eval-every` | 0 | 仅 steps 模式可按 N 个 optimizer step 验证；epochs 模式只在 epoch 结束验证 |
| `--initial-eval` | 关闭 | epoch 模式可显式增加一次训练前验证；不参与 best 选择 |
| `--skip-initial-eval` | 关闭 | 跳过 steps 模式保留的默认训练前验证 |
| `--validation-role` | `validation` | `test` 只在最终评估一次，不用于 checkpoint 选择 |
| `--no-tensorboard` | 关闭 | 关闭 TensorBoard 事件写入，仍保留 JSONL 日志 |
| `--strict-lengths` | 关闭 | 任何输入或答案长度超限时直接报错 |

`final_only` 复用原 LLaDA 的 block/norm 模块及注入的 LoRA，不改 checkpoint 参数名。
旧路径 `--readout all_logits` 保留完整词表计算，作为数值对照。
两条路径的 Transformer 次数相同；省掉中间 LM head 不意味着 Transformer 计算消失。
训练使用 non-reentrant activation checkpointing，反向时会重算部分前向，以计算换显存。

**与普通 LLaDA SFT 的区别：**这里没有随机 mask 比例，answer 始终全 MASK，
最终一轮同位置 CE 监督；也不是 block32 / generation256 的原生扩散解码。
目前固定轮次，无 adaptive stopping、KV cache 或 dual cache。

单轮初步控制可以使用相同数据、seed、其他配置，仅改 `--rounds 1`。
它仍有相同的额外槽位，既不是标准 diffusion SFT，也不是等算力控制；
后续需要更多消融才能将收益归因于 latent reasoning。

## 7. 输出、重载与限制

每次运行保存：

```text
runs/<name>/
  config.json          # 本次参数
  environment.json     # 版本、实际样本数、参数数和权重加载检查
  metrics.jsonl        # 训练 CE、梯度范数、时间、显存和验证结果
  tensorboard/         # TensorBoard events；每次运行使用独立目录
  sample_order.jsonl   # 每次 optimizer update 实际使用的训练行索引
  data_coverage.json   # 训练结束后的样本覆盖统计
  predictions/         # 每次验证的逐题预测，文件名为 step000000.jsonl 等
  adapter/             # 最近完成 epoch／steps 训练的 LoRA 与 tokenizer
  last_checkpoint.json # 最新 LoRA 对应的 epoch、step 与样本数
  best_adapter/        # 独立验证集最低 val_ce 对应的 LoRA 与 tokenizer
  best_checkpoint.json # 最佳 checkpoint 指标、epoch、step 与数据来源
  reload_check.json    # 使用 --verify-reload 时保存
```

验证指标为 CE 和简化的答案字符串 EM：answer 位置并行 argmax，在第一个 EOS 截断，
去掉逗号后比较。它不是官方 GSM8K evaluation harness。
epoch 模式每个 epoch 验证一次，最后一个 epoch 不重复验证。steps 模式的实现检查可显式使用 `--eval-every N`，
最后一步即使恰好也是周期验证点，也只验证一次。`--validation-role test` 始终只做最终评估，不生成最佳 checkpoint。
最近完成的 adapter 在评估前保存；即使后续评估失败，已经训练出的 adapter 仍会保留。
每次验证保存逐题 gold、prediction、CE、字符串 exact match、完整预测 token IDs 和 EOS 是否出现／位置，
并记录 `val_eos_rate` 与 `eval_s`，便于区分数值答案出错和输出格式问题。
还记录有效答案 token 的准确率、token 加权 CE、预测置信度和熵；这些统计复用本次验证 logits，不增加一次模型 forward。
`example_index` 对应经过长度筛选后、按输入文件顺序排列的验证样本。

训练指标中的 `samples_seen` 是已处理的样本总数，`microbatches` 是本次 update 的实际累积条数。
epoch 模式还记录 `epoch` 和 `epoch_samples_seen`。
`sample_order.jsonl` 逐次记录 `{step, epoch, indices}`，其中 `indices` 是被接受训练行从 0 开始的索引；
开启 `--strict-lengths` 且不设置 `--train-limit` 时，它们直接对应完整输入文件的行号。
结束后的 `data_coverage.json` 给出各 epoch 的唯一样本数和 `complete` 标志，便于核对是否遍历全量。

`step_train_s` 是本次 optimizer step 的训练时间，`elapsed_train_s` 是这些时间的累计，
均不包括验证、模型加载、保存或日志写入；验证耗时单独记录在 `eval_s`。
`allocated_gib`／`reserved_gib` 是每步结束时的 PyTorch 当前分配／缓存池显存，
`peak_allocated_gib` 是训练开始时重置计数后至当前的历史分配峰值（包含中途验证），不代表当前占用。
这些显存数值不包含 CUDA context 等外部分配。每步训练 CE 来自当步累积的样本，
不是固定训练子集的评估结果；epoch 模式使用无放回排列，steps 模式使用有放回抽样。

`--verify-reload` 检查保存前后的 adapter 权重和 logits 是否完全相同。
当前 PEFT 版本向已有 BF16 模型添加 adapter 时可能先舍入权重；本项目的
`load_fp32_adapter()` 在加载前创建 FP32 目标参数，避免该路径的精度丢失。
跨机器使用 adapter 时，必须保留相同底座、tokenizer、slot 数、rounds 和递归流程；
不要直接用普通 `model.generate()` 代替这里的 `recurrent_logits()`。
当前没有 optimizer 状态续训功能。

加载最佳结果时，使用 `best_checkpoint.json` 对应运行的配置与 `best_adapter/`；
`adapter/` 是最后结果，两者可能来自不同 epoch。可以复用 `load_fp32_adapter()` 加载最佳 LoRA：

```python
from transformers import AutoModel
from experiment import add_lora, load_fp32_adapter
import torch

base = AutoModel.from_pretrained(
    "checkpoints/llada15", trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16,
)
model = add_lora(base, rank=8)  # 使用该次训练 config.json 中的 rank
load_fp32_adapter(model, "runs/gsm8k-k2-seed42-e3/best_adapter", adapter_name="best")
model = model.to("cuda:0").eval()
# 后续使用同一运行配置的 recurrent_logits()，不要改成普通 generate()。
```

两个 adapter 目录只保存 LoRA 与 tokenizer；共享盘中的 8B 底座仍直接引用，不另外复制。

### TensorBoard：查看训练和每个 epoch 的验证结果

在实验机器的项目目录、已安装依赖的环境中运行：

```bash
python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6007
```

在自己电脑建立 SSH 隧道，然后打开 `http://127.0.0.1:6007`：

```bash
ssh -N -L 6007:127.0.0.1:6007 user@your-training-machine
```

默认横轴为 optimizer step，训练每步记录，验证曲线只在 epoch 边界出现。
TensorBoard 的 smoothing 只影响显示，不改变 `val_ce` 或 checkpoint 选择。

| 标签 | 观察内容 |
| --- | --- |
| `train/loss`、`train/token_loss`、`train/epoch_loss` | 当步 CE、token 加权 CE、整轮训练平均 CE |
| `val/loss`、`val/token_loss` | 最佳 checkpoint 使用的逐题平均 CE、token 加权 CE |
| `val/answer_em`、`val/token_accuracy`、`val/eos_rate` | 最终数字答案、token 与输出终止符表现 |
| `val/mean_confidence`、`val/mean_entropy` | 验证预测的置信度和不确定性 |
| `val/latent_rms`、`val/latent_round_cosine`、`val/latent_relative_change` | hidden 幅度与相邻轮次变化；K1 没有跨轮指标 |
| `optimizer/learning_rate`、`optimizer/grad_norm` | 学习率、裁剪前梯度范数 |
| `performance/*`、`gpu/*`、`progress/*` | 训练／验证耗时、吞吐、显存、样本数与 epoch 进度 |

运行参数也写入 TensorBoard 的文本面板。`metrics.jsonl` 仍是原始指标来源，逐题预测留在 `predictions/`。
这里的吞吐受共享 GPU 上其他进程影响，不能直接作为单进程效率对照。
latent 的相似度和变化幅度仅作诊断，不能单独证明推理正确或已经收敛。
`performance/model_positions_per_s` 是每秒处理的前向位置数（计入 rounds），不是生成 token 速度或完整 FLOPs。

已经结束、只有 JSONL 日志的旧实验可补导出，无需重跑训练。替换为实际运行目录：

```bash
bash run.sh export_tensorboard.py runs/old-full-k1 runs/old-full-k2 \
  --output runs/tensorboard_history --eval-split test
```

`--eval-split auto` 会检查原配置及数据来源；历史全量运行虽然把指标写成 `val_*`，
实际上评估的是官方 test，导出后显示为 `test/*`。独立验证集运行使用 `val/*`。
导出只转换已有信息，不补造缺失的 epoch 指标或历史最佳 checkpoint；重复导出相同输入不会叠加重复事件。
旧日志若没有 `wall_time`，事件时间使用导出时间；请用 STEP 横轴和已记录的 `elapsed_train_s`，
不要把旧日志的 TensorBoard WALL 时间当作原始训练时间。

`final_only` 只覆盖当前**单设备、batch1 无 padding、RoPE、block_group_size=1** 的 LLaDA 路径。
没有验证 Dream、iLLaDA、LLaDA2、FSDP、自动 device-map 或模型量化。
错误传播、停止策略和实际推理收益均仍是实验问题。

## 8. 文件说明与参考

- `experiment.py`：LoRA、递归、答案 loss、训练与评估，以及可选的小子集准备。
- `prepare_gsm8k.py`：准备官方原版 GSM8K 全量 train/test，不使用 GSM8K-Aug。
- `telemetry.py` / `export_tensorboard.py`：实时 TensorBoard 日志和历史 JSONL 指标导出。
- `llada_backbone.py`：不经过词表 head 的原模型 backbone，以及最终 answer 投影。
- `prepare_checkpoint.py`：模型下载／本地准备、代码补全与分片检查。
- `make_tiny_checkpoint.py`：用真实 tokenizer 创建随机小模型。
- `check_training.py` / `check_readout.py` / `check_data.py`：实现和数据检查。
- `check_epochs.py`：检查 epoch 完整覆盖、最后一个不足 4 条的更新、最终评估和旧 steps 模式；使用随机小模型和可选调试子集。
- `check_validation.py`：检查每轮验证、best/latest 权重对应关系、指标计算、测试集隔离和保存失败回滚。
- `docs/ar_implementation_notes.md`：Coconut、CODI、CoLaR 官方代码核对。

参考：[Coconut](https://github.com/facebookresearch/coconut)、
[CODI](https://github.com/zhenyi4/codi)、[CoLaR](https://github.com/xiaomi-research/colar)、
[LLaDA-1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5)。

本项目借鉴连续状态回填的思路，当前并未实现 CODI 蒸馏或 CoLaR 压缩监督。
