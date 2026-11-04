# AR 实现核对（2026-11-05）

“latent 不输出文本”和“代码不计算词表 logits”不同。以下结论来自官方源码，
不是对论文概念的推测。本仓库不包含这些项目的代码副本。

| 方法 | 连续状态反馈 | 训练 | 公开生成／latent 实现 |
| --- | --- | --- | --- |
| Coconut | 前一位置最后层 hidden 直接回填下一位置，不 detach | 课程训练、文本 CE | latent 阶段复用 causal KV cache，但仍调用完整 causal-LM，计算 logits |
| CODI | hidden 回填；可选 MLP + LayerNorm projector | 学生答案 CE、teacher CE、答案边界的多层 hidden 蒸馏 | 固定 latent 步数、causal KV cache，仍调用完整 causal-LM |
| CoLaR | latent head 预测 embedding 分布，采样并缩放后回填 | SFT 用真实 CoT 的压缩 embedding 轨迹作监督，也支持 RL | 仍计算词表 logits，通过采样出的 thinking separator 停止或达到步数上限 |

源码链接：

- [Coconut coconut.py](https://github.com/facebookresearch/coconut/blob/main/coconut.py)
- [CODI src/model.py](https://github.com/zhenyi4/codi/blob/main/src/model.py)
- [CODI test.py](https://github.com/zhenyi4/codi/blob/main/test.py)
- [CoLaR colar.py](https://github.com/xiaomi-research/colar/blob/main/src/models/colar.py)
- [CoLaR model_base.py](https://github.com/xiaomi-research/colar/blob/main/src/models/model_base.py)
- [CoLaR projector.py](https://github.com/xiaomi-research/colar/blob/main/src/modules/projector.py)
- [CoLaR 默认配置](https://github.com/xiaomi-research/colar/blob/main/src/configs/models/colar.yaml)

CoLaR SFT 将真实 CoT 的连续 token embeddings 聚合为压缩轨迹，并训练下一 embedding 的预测。
这不同于对当前词表预测概率做加权的 soft-thinking。默认配置使用 MSE latent loss 与文本 CE，
`pred_embed_forward_weight=0`，并非每次训练都展开完整的预测 latent 链。

CODI 的学生链保留梯度；蒸馏目标在 no_grad 下计算并 detach，teacher CE 则另有带梯度前向。
Coconut 官方 `generate()` 的答案循环还会重新传完整输入，不能把全部生成阶段都视为已充分 cache 优化。

## 移植到 masked LLaDA

保留连续状态回填、学生链跨轮梯度和最终答案监督。AR 的 causal cache 不能原样迁移：
双向模型的旧位置表示也可能受更新后的 latent 影响。AR 的答案 teacher forcing 也不能直接
放进全双向上下文，否则 latent 能看到答案。因此本实验每轮 answer 输入始终是 MASK。

本项目的 `final_only` 路径在中间轮次不计算词表投影，最后只对 answer hidden 调用 LM head。
Transformer 迭代次数没有增加。没有照搬 CoLaR 依赖词表 logits 的停止规则，暂时固定轮次。
RMS 幅度匹配是本项目的实验选择，不应称为原样复现 Coconut／CODI／CoLaR。

小模型数值检查已经通过；真实 8B 模型的开销和推理效果仍需独立实验。
