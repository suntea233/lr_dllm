# lr_dllm

在 **masked LLaDA** 上实验连续 hidden-state reasoning：reasoning 位置不解码成 token，
在 iteration 之间回填 hidden，最后只对 answer 位置做词表投影。底座冻结，使用 LoRA 训练。

**当前是实现原型。** 已通过小型 LLaDA 的 CPU/CUDA 前向、反向、数值一致性和 adapter 重载检查；
尚未完成预训练 LLaDA-1.5 8B 的完整训练验证，尚无 GSM8K 效果结论。
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

## 3. 准备 GSM8K 小数据

```bash
bash run.sh experiment.py prepare
bash run.sh check_data.py
```

从 `openai/gsm8k` 的官方 **train** 划分中取数据，按 question 去重，以 seed42 打乱：

| 文件 | 数量 | 说明 |
| --- | ---: | --- |
| `data/train64.jsonl` | 64 | 实现检查／快速过拟合实验 |
| `data/train1000.jsonl` | 1000 | 稍大规模迭代，与 train64 为嵌套关系 |
| `data/val128.jsonl` | 128 | 与上述训练集无重复问题 |

仅取原始答案中 `####` 后的最终答案；不把 CoT 放入训练输入，不使用官方 test 调参。
`data/manifest.json` 记录来源、seed 和样本数；每条记录保留原始行号。
`check_data.py` 使用真实 tokenizer 检查长度、划分和文件 hash。

已有缓存 Arrow 文件时，可以离线准备：

```bash
bash run.sh experiment.py prepare --arrow /path/to/gsm8k-train.arrow
```

`--max-prompt 256` 和 `--answer-slots 16` 超限样本会被跳过并打印数量，不静默截断。
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

## 5. 准备完整 LLaDA-1.5 并尝试 8B LoRA

### 从 Hugging Face 下载

```bash
# 不加 --metadata-only，会下载完整权重（约 16 GB）
bash run.sh prepare_checkpoint.py
```

若已执行 metadata-only，可在同一目录补齐权重。下载文件默认位于 Hugging Face 缓存，
`checkpoints/llada15` 的权重文件链接到缓存；请保留缓存。
要在输出目录保存独立权重副本，首次准备时加 `--copy-weights` 并选一个新目录。

### 使用其他机器上已经存在的模型

```bash
bash run.sh prepare_checkpoint.py \
  --source /path/to/LLaDA-1.5 \
  --output checkpoints/llada15-local
```

若源目录缺少 `modeling_llada.py` / `configuration_llada.py`，脚本会在架构配置匹配后，
从 `GSAI-ML/LLaDA-8B-Instruct` 补入模型代码。离线时可加：
`--code-source /path/to/LLaDA-8B-Instruct`。
完整权重索引和分片存在性会被检查，加载时还会检查参数是否完整匹配。
来源、revision 和代码 hash 写入输出目录的 `provenance.json`。
固定实验建议为下载指定 `--revision <commit>`，必要时也指定 `--code-revision <commit>`。

### 先运行一个 optimizer step

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh experiment.py train \
  --model checkpoints/llada15 \
  --data data/train64.jsonl --validation data/val128.jsonl \
  --output runs/llada15-one-step \
  --train-limit 1 --eval-limit 1 --steps 1 --accumulation 1 \
  --rounds 2 --latent-slots 16 --answer-slots 16 \
  --rank 8 --dtype bfloat16 --verify-reload
```

确认实际显存、时间和数值正常后，再做 100 步短训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh experiment.py train \
  --model checkpoints/llada15 \
  --data data/train64.jsonl --validation data/val128.jsonl \
  --output runs/llada15-k2-seed42 \
  --steps 100 --accumulation 4 --rounds 2 \
  --latent-slots 16 --answer-slots 16 --rank 8 --lr 1e-4 \
  --dtype bfloat16 --seed 42 --verify-reload
```

默认空闲显存门槛是 `--min-free-gib 40`，这是保守的启动保护，**不是实测最低需求**，
也不会预留显存。8B BF16 底座本身约 15 GiB，LoRA 仍需要底座和跨轮激活。
当前没有量化、CPU offload 或多卡分片支持；不要把多张卡的空闲显存简单相加。
本机此前的 8B 验证受空闲显存和共享存储速度限制，不能提供可靠的训练时间估计。

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
| `--readout` | `final_only` | 中间跳过 LM head，最后只投影 answer |

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
  adapter/             # LoRA adapter 与 tokenizer，不含完整底座
  reload_check.json    # 使用 --verify-reload 时保存
```

验证指标为 CE 和简化的答案字符串 EM：answer 位置并行 argmax，在第一个 EOS 截断，
去掉逗号后比较。它不是官方 GSM8K evaluation harness。
训练耗时不包括初始验证、模型加载和保存；显存日志是 PyTorch allocated 峰值。

`--verify-reload` 检查保存前后的 adapter 权重和 logits 是否完全相同。
当前 PEFT 版本向已有 BF16 模型添加 adapter 时可能先舍入权重；本项目的
`load_fp32_adapter()` 在加载前创建 FP32 目标参数，避免该路径的精度丢失。
跨机器使用 adapter 时，必须保留相同底座、tokenizer、slot 数、rounds 和递归流程；
不要直接用普通 `model.generate()` 代替这里的 `recurrent_logits()`。
当前没有 optimizer 状态续训功能。

`final_only` 只覆盖当前**单设备、batch1 无 padding、RoPE、block_group_size=1** 的 LLaDA 路径。
没有验证 Dream、iLLaDA、LLaDA2、FSDP、自动 device-map 或模型量化。
错误传播、停止策略和实际推理收益均仍是实验问题。

## 8. 文件说明与参考

- `experiment.py`：数据准备、LoRA、递归、答案 loss、训练与验证。
- `llada_backbone.py`：不经过词表 head 的原模型 backbone，以及最终 answer 投影。
- `prepare_checkpoint.py`：模型下载／本地准备、代码补全与分片检查。
- `make_tiny_checkpoint.py`：用真实 tokenizer 创建随机小模型。
- `check_training.py` / `check_readout.py` / `check_data.py`：实现和数据检查。
- `docs/ar_implementation_notes.md`：Coconut、CODI、CoLaR 官方代码核对。

参考：[Coconut](https://github.com/facebookresearch/coconut)、
[CODI](https://github.com/zhenyi4/codi)、[CoLaR](https://github.com/xiaomi-research/colar)、
[LLaDA-1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5)。

本项目借鉴连续状态回填的思路，当前并未实现 CODI 蒸馏或 CoLaR 压缩监督。
