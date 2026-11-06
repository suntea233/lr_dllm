# lr_dllm

在 **masked LLaDA** 上实验连续 hidden-state reasoning：reasoning 位置不解码成 token，
在 iteration 之间回填 hidden，最后只对 answer 位置做词表投影。底座冻结，使用 LoRA 训练。

**当前 K8 是全局 hidden 迭代，不是 block-AR。** 第一次 forward 后，全部240个 reasoning 位置
同时回填 hidden，共执行固定8轮；16个答案输入始终为MASK，最后一轮才读出答案。
该模式中的`block_size=16`只记录参考布局，不控制计算，也没有逐块完成判定。
另已实现用户确认的独立`block_confidence`试跑模式：块内所有位置的置信度超过0.9才切块，
每块最多8轮，已完成块保存连续hidden；具体协议和限制见第11节。它不改变正在运行的全局K8。
原版 GSM8K 的 CoLaR 6726/747 划分上，K1/K2 已完成3 epochs，修正后的K8仍在训练；
这里的运行状态截至2026-11-06，并不代表已验证推理收益。

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
python serve_tensorboard.py --host 127.0.0.1 --port 6007
```

显示名称和事件目录的对应关系保存在 `tensorboard_runs.json`，启动脚本在 `.tensorboard_views/`
中创建目录链接作为显示别名。原始运行目录、checkpoint 和事件文件保留原位，运行中的训练继续写入。
当前页面按下面类别显示，小模型检查和显存探测不加入实验列表：

| 分组 | 含义 | 当前 K2 示例 |
| --- | --- | --- |
| `正式训练` | 6726 train / 747 独立 val，原始训练曲线 | `K2_GSM8K90-10_3epoch_20261105` |
| `历史实验` | 更早的独立实验：7473 train / 1319 官方 test，1 epoch，旧日志导入 | `K2_GSM8K全量_官方test_1epoch_20261105` |
| `诊断快照` | 正式训练日志的派生统计，并非一次新训练 | `K2_同次正式训练_100步统计_仅前2epoch` |
| `已停止` | 配置不符原意而停止的运行，只保留追溯记录 | `K8_误加渐进回填_168step_20261106` |

K2 诊断快照生成于训练中途：源日志当时到 step3423，100 步窗口到3400，只有 epoch1/2 的验证诊断，
不包含最后一轮，也不自动更新。K1 诊断覆盖三个 epoch，其100步窗口到5000。
判断正式训练结果请看 `正式训练` 中的原始 `train/*`、`val/*`；历史实验对应的是 `test/*`。
K8 名称额外注明全位置 hidden、hidden240 和4-shot，避免把它与 K1/K2 的 hidden16、0-shot 当作同一配置。

后续显示名使用 `类别/K值_数据划分_区别设置_日期`；相同配置的重复运行补充 seed 或具体时间。
GPU 编号仅作为资源记录，完整参数保存在运行的 `config.json` 和清单的 `description`。
新增实验时在清单中添加 `display_name`、`events` 和说明，重启 TensorBoard 服务即可生效。
排队的实验可设置`allow_pending: true`：只创建显示用链接，不提前创建训练输出目录，
worker实际写入事件后会自动出现曲线。没有曲线不代表已完成训练，等待状态以launcher/state.json为准。
其他机器需填写当地实际存在的事件目录，也可通过 `--manifest /path/to/runs.json` 指定独立清单。

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

`final_only` 覆盖**单设备或显式连续层分片、batch1 无 padding、RoPE、block_group_size=1** 的 LLaDA 路径。
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

## 9. 显式两卡模型分片与后台等待

单卡显存不足时，可以在同一个进程中把连续 Transformer 层分到两张 CUDA 卡：

```bash
CUDA_VISIBLE_DEVICES=2,3 bash run.sh experiment.py train \
  --model checkpoints/llada15-local \
  --data data/gsm8k_colar_split/train.jsonl \
  --validation data/gsm8k_colar_split/val.jsonl \
  --output runs/my-k8-model-parallel \
  --model-parallel-devices cuda:0,cuda:1 \
  --rounds 8 --epochs 3 --strict-lengths
```

这只是现有 hidden recurrence 的分片示例，**没有启用原生 block 采样**。
embedding 在首卡，输出 norm/head 在末卡，激活在层边界显式跨卡复制并保留梯度。
底座权重不复制；这不是 DDP，也不是吞吐并行。仅支持 `--readout final_only`，
不支持自动 device-map/offload、跨卡 tied embeddings 或非零 dropout。
`--min-free-gib` 和 `--max-memory-gib` 均按每张卡解释，后者不包含 CUDA context。
JSON 和 TensorBoard 同时记录各卡显存；总 peak 指各卡历史 peak 的和，并非精确的同时峰值。

`check_model_parallel.py` 默认跑 CPU；显式传入两张卡才检查真实设备间复制。
已经通过 FP32/BF16、K=1/2/8、checkpoint 开关的 logits/全部 LoRA 梯度一致性，
以及 optimizer 更新、adapter 精确重载检查。这些小模型检查不代表 8B 的显存或速度结果。

```bash
CUDA_VISIBLE_DEVICES=2,3 bash run.sh check_model_parallel.py --devices cuda:0 cuda:1
```

`launch_when_free.py --spec /abs/job.json --output /abs/new-launch-directory` 支持等待一张或两张卡。
spec 包含 `argv`（参数列表，非 shell 字符串）、`cwd`、`env`（仅显式覆盖项）、
`candidate_gpus`、`gpu_count`、`min_free_gib`，可设置 `poll_seconds`、`stable_polls`、
`max_wait_seconds`。默认每 30 秒查询，连续 3 次满足显存条件才启动，最多等待 24 小时。
它为 worker 设置所选 GPU UUID 的 `CUDA_VISIBLE_DEVICES`，并记录状态、PID、退出码和日志。
不会停止其他任务；失败不会自动反复重试。同一个启动目录不能重复使用。
启动器本身也应以 `start_new_session=True` 或 nohup 方式运行；外层日志不要指向
它自行创建的 `supervisor.log`。`check_launch_when_free.py` 用 CPU mock 检查这些行为。

## 10. 官方 LLaDA-1.5 配置与本项目 K 的区别

固定参考 [ML-GSAI/LLaDA commit 9182493 的 GSM8K 配置](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/opencompass/examples/llada_1p5_gen_gsm8k_length256_block16_confidence.py)：
原生设置为 4-shot CoT、256 个起始 MASK、block size 16、全序列 256 个采样步骤、
temperature 0、CFG 0、low-confidence remasking；EOS logits 保留，EOS/EOT confidence 被抑制。
官方实现不使用 dual cache。训练的随机 response masking 与推理的 block schedule 是两回事。

本项目 `--rounds 8` 是全 canvas **8 次 hidden forward**。原生 256/16 则有 16 个顺序 block，
全局 steps=8 不合法；每 block 8 步意味着全局 128 次 forward。仅增大 latent slots，
不能称为实现了原生 block16 采样。

官方四个示例中的 Angelo/Melanie 问题位于当前 CoLaR 验证集 source_index 7364，
因此当前验证集不能直接复用那一套固定示例；使用 4-shot 时需选训练集示例并记录其来源。

### 本轮纯 hidden K=8：256 个位置、从第二轮起全位置回填

本轮将生成区域固定为 **240 个 latent 位置 + 16 个 answer 位置 = 256**，
从训练第一步起固定执行 **8 次全局 backbone forward**。
第一轮后，全部 240 个 reasoning 位置都接收上一轮的 hidden，没有逐块开放或重置为 MASK：

| 全局 forward | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 输入中回填 hidden 的 latent 位置数 | 0 | 240 | 240 | 240 | 240 | 240 | 240 | 240 |
| 输入中仍为 MASK embedding 的位置数（均为初始化或 answer） | 256 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

每轮重新计算整个序列，所有 latent 位置同时更新并参与双向 attention。
上一轮的全部 latent hidden 经 RMS 缩放后回填到下一轮，
reasoning 位置始终不采样或提交 token。16 个 answer 输入始终保持 MASK，
仅第 8 轮对 answer hidden 执行词表 head；监督为答案数值和 EOS 的交叉熵，
其余 answer padding 不计 loss。训练使用 activation checkpointing，反向传播会重算部分 forward，
因此这里的 K=8 描述递归轮数，不包含反向重算次数。

`--block-size 16` 仅记录官方参考布局并检查位置数整除关系，不参与 hidden 更新调度，
也不减少每轮计算量。当前方法没有原生 confidence unmasking、显式 CoT 或 dual cache；
K=8 始终指全序列8轮。此前逐步开放 hidden 前缀的版本误加了用户未要求的限制，
该运行已停止并单独标记；其只读源码快照和日志保留用于追溯，不能作为当前方法的结果。

训练使用原版 GSM8K 的 CoLaR 划分：**6726 条 train、747 条 val**，不使用 GSM8K-Aug。
训练 prompt 仍为单题，配置名 `--prompt-profile answer_only` 表示提示模型只输出最终答案；
模型内部仍执行上述 K=8 hidden reasoning，这个名称不表示新增一个无 latent baseline。
验证使用 `--eval-prompt-profile llada4shot`，示例固定取自训练集
source_index `[669, 1708, 6724, 0, 1]`，每题取其中前四个不同于当前题的示例。
示例保留完整原始解答（含 `####`），验证集与该池无问题重叠；当前题不包含 gold answer。
四个示例与当前问题拼入一条 user message 后应用 chat template，属于本项目的 four-shot 适配。

当前 tokenizer 实测 747 条验证 prompt 最长为 **1459 tokens**，
设置 `--max-eval-prompt 1536`；最长验证总序列为 1459 + 256 = 1715 tokens。
使用 `--strict-lengths` 检查长度，不截断或静默丢弃题目。每个 epoch 结束时才计算验证指标，
按最低 `val_ce`（逐题平均）保存 `best_adapter/` 和 `best_checkpoint.json`，并记录 TensorBoard。
**本轮 four-shot 验证与旧 K=1/K=2 的单题验证不同，分数变化不能直接归因于 K**；
比较 K 的影响时必须统一 prompt、长度、block 规则和数据划分。

### 准备独立运行目录与后台等待

campaign helper 面向共享盘部署：它要求权重分片以 symlink 引用 `--shared-model-root`，
并核对完整的 CoLaR 数据 manifest。下面两个模型路径是本机示例，其他机器应替换为实际路径：

```bash
python prepare_k8_campaign.py \
  --campaign runs/my-k8 \
  --model checkpoints/llada15-local \
  --shared-model-root /mnt/public/models/LLaDA-1.5 \
  --gpus 0 2 \
  --min-free-gib 15 \
  --cap-gib 13.5 \
  --max-eval-prompt 1536
```

该命令只准备新目录和启动配置，保存代码、数据及来源记录，复用现有模型权重，
不会启动训练或复制权重。默认配置为 K=8、240/16 位置、全位置hidden回填、参考block16、3 epochs、LoRA rank8、
learning rate `1e-4`、梯度累积为 4，训练单题、验证 four-shot。
`--gpus` 指物理 GPU 编号或 UUID，训练使用两卡模型分片。
这里的每卡空闲阈值 15 GiB 和 allocator cap 13.5 GiB **只是命令示例，必须根据实测修改**；
需同时检查训练反向传播及最长 four-shot 验证的显存，allocator cap 不包含 CUDA context。

检查生成的 `campaign_plan.json` 和 `launch_spec.json` 后，可将等待进程挂到后台：

```bash
nohup python -u runs/my-k8/source/launch_when_free.py \
  --spec "$PWD/runs/my-k8/launch_spec.json" \
  --output "$PWD/runs/my-k8/launcher" \
  > runs/my-k8/nohup.log 2>&1 < /dev/null &
```

等待器按配置检查两张卡的空闲显存，满足连续检查条件后启动训练，
训练输出位于 `runs/my-k8/train`。每次复现实验使用新的 campaign 目录；
保留已有 launcher 目录，以防同一配置被重复启动。

如果使用前文通用流程准备的 `checkpoints/llada15`，可直接调用训练入口，
不依赖上述共享盘目录结构。先完成原版 GSM8K CoLaR 划分，再按实际空闲卡调整 GPU 编号和显存参数：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run.sh experiment.py train \
  --model checkpoints/llada15 \
  --data data/gsm8k_colar_split/train.jsonl \
  --validation data/gsm8k_colar_split/val.jsonl \
  --output runs/k8-all-hidden-seed42 \
  --model-parallel-devices cuda:0,cuda:1 \
  --rounds 8 --latent-slots 240 --answer-slots 16 \
  --generation-length 256 --block-size 16 \
  --prompt-profile answer_only --eval-prompt-profile llada4shot \
  --max-prompt 256 --max-eval-prompt 1536 \
  --epochs 3 --accumulation 4 --rank 8 --lr 1e-4 --seed 42 \
  --dtype bfloat16 --readout final_only \
  --skip-initial-eval --strict-lengths --verify-reload \
  --min-free-gib 14 --max-memory-gib 12.5
```

这组显存数字来自当前两卡短测，仅是复现示例，需按当地负载检查。
以上命令仍是全局hidden迭代，没有启用block-AR或自适应停止。

### 完成判定的含义与设计边界

默认global模式只按固定全局K轮数结束，没有“第一个block已经推理完成”的判断。
latent block是固定数量的位置，不自动对应一道算式或一个语义推理步骤，
当前answer+EOS监督也没有训练任何block完成标签。

| 可讨论的规则 | 实际判断的内容 | 需要补充的设计 |
| --- | --- | --- |
| 每块固定迭代预算 | 分配的计算已经用完，并不证明推理完成 | K按全局还是每块计数、各块预算和最终答案读出预算 |
| hidden变化量阈值 | 相邻轮次表示变化较小 | 独立验证集校准、最小轮数、连续满足次数、最大预算；表示稳定可能只是错误固定点 |
| learned halt head | 模型预测继续计算是否有价值 | 完成/收益监督或RL奖励、额外head的训练方式、误停代价；当前没有这些组件 |

hidden本身没有词表概率，因此不能直接使用“所有reasoning位置token置信度很高”的判据。
即使额外经过LM head计算概率，也需验证其与完成状态的关系；高置信度不保证答案正确。
本项目当前仅在末轮投影答案，增加逐轮置信度探测也会改变计算开销。

若引入块间顺序，还必须先确定：已处理块的持久hidden是否冻结、是否允许后续块回改、
未来块是否参与attention、如何划定块边界以及训练时如何展开反传。
冻结早期hidden同样会限制后续修正；允许回改会改变严格单向block-AR的语义。
240个reasoning位置按16分块共有15块；如果每块8轮，reasoning预算就是120次forward，
还未计入最终答案读出，不能与当前全局8轮混称K8。

上述收敛阈值和learned halt head仍未实现。用户后来确认了独立的词表置信度规则，
已作为第11节的显式实验模式实现；它是切块启发式，不是推理正确性的证明。
改变块间顺序、冻结策略、停止条件、训练监督或K的定义前，应先明确方案与计算代价，
由实验负责人确认后再更改运行配置。

## 11. 独立试跑：按置信度结束连续 hidden block

用户确认的协议为：**阈值0.9、block16、每块最多8轮、完成后固定持久hidden、训练与推理同规则**。
使用`--reasoning-mode block_confidence`显式启用；该模式拒绝`--rounds`，
通过`--min-block-rounds 1 --max-block-rounds 8`表达每块预算。

1. 第j块的输入为固定prompt embeddings、已完成块保存的hidden和当前块；未来块不进入输入。
   当前块第一轮输入MASK，以后全部位置输入上一轮经RMS缩放的hidden，不重新MASK任何位置。
2. 当前块每轮输出的原始final-norm hidden经过原生LM head，使用FP32 softmax、temperature=1。
   只有`min_position(max_vocab(probability)) > 0.9`才按置信度切块，比较为严格大于。
   原词表中的EOS/MASK等特殊token不屏蔽，额外记录其占比以识别虚假高置信度。
3. 达到8轮仍未满足条件时按预算强制切块，日志区分`confidence`和`max_rounds`。
   完成块保存的是本轮经RMS缩放的连续hidden，不回填argmax token，不作soft embedding混合。
4. 已保存的持久hidden作为后续块的固定输入，不被回写，也**不detach**；答案loss可跨块反传。
   backbone仍会重新计算整个已提供前缀的中间表示，这些表示可看当前块；本版没有KV cache，
   没有额外的block-causal attention mask。块间顺序由未来块不进入输入保证。
5. 所有15个reasoning块完成后，才加入16个答案MASK，并额外执行一次backbone和答案head。
   监督仍是数值答案+EOS CE；confidence probe运行在no_grad下，不增加辅助loss或可微停止奖励。

每例至少16次、最多121次backbone forward，以及15至120次confidence head和1次answer head。
这些计数不含checkpoint在反向传播中的重算，不等同于完整FLOPs或wall time。
固定前块hidden会限制后续修正；高词表置信度也可能对应错误答案或特殊token塌缩。

训练入口的关键参数如下，其他模型、数据、LoRA、显存参数按前文完整命令填写：

```bash
--reasoning-mode block_confidence \
--latent-slots 240 --answer-slots 16 --generation-length 256 --block-size 16 \
--confidence-threshold 0.9 --min-block-rounds 1 --max-block-rounds 8
```

`reasoning_steps.jsonl`逐条记录训练rollout的各块轮数、停止原因和每轮confidence，
验证的相同信息写入`predictions/`。TensorBoard记录`train/`与`val/`下的
`actual_backbone_calls_mean`、`block_rounds_mean`、`blocks_confidence_fraction`、
`blocks_forced_fraction`、`block_final_min_confidence_mean`、`probe_special_top1_fraction`。
block模式的latent相似度仅比较同一块内相邻轮次，不比较不同块。

CPU检查：`CUDA_VISIBLE_DEVICES='' bash run.sh check_block_reasoning.py`。
已覆盖实际输入、每个位置的梯度、严格阈值、强制停止、checkpoint一致性，
以及15块每块8轮的121次forward路径。随机小模型通过不代表8B模型有效。

本机可用`prepare_block_probe.py --campaign runs/block-confidence-probe`准备独立8B技术短测。
该helper默认引用本机现有的两条最长train probe和两条最长CoLaR val probe；
其他机器需显式传入`--train-probe`和`--validation-probe`，且每个文件必须是相应划分的两条原始记录。
模型路径及`--shared-model-root`同样应按本机共享盘调整，不复制底座权重。
默认候选GPU为1、3、4，每卡空闲门槛16GiB、allocator cap14GiB；这些是待实测保护值。
传入`--gpus 0 1 2 3 4`可在全部卡中等待任意两张满足门槛的卡，不会停止已有进程。
helper只准备配置，启动`campaign_plan.json`中的launcher后才等待显存并运行。
短测只做1次参数更新（梯度累积4）、验证2条及adapter重载，不自动接续全量训练，
不能把短测的loss或准确率当成GSM8K正式结果。
