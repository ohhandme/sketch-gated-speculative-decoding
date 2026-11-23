# 云边投机推理的低秩上行与草图门控

这是 [实验规格](docs/program.md) 的单机 CPU 参考实现。默认在本地随机初始化两个 tiny Qwen3（草稿 2 层，目标 3 层，词表 2048）。默认配置无需下载模型。`su/models.py` 也支持本地 Hugging Face 权重目录，两个模型需共用 token ID 语义。

## 复跑

在本目录使用 Python 3.12，在已安装依赖的环境中执行。`requirements.txt` 包含运行和单测依赖。

```powershell
python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m unittest discover -s tests -v
python -m su.cli all --config configs/default.yaml --steps 8 --out runs/reproduce
```

`all` 运行四路基线、小规模独立校准、五个带宽档和四组消融。`run --mode ours --theta 0` 可以强制测试非零偏差触发回退，逐步 token、字节和计时写入 `run.json`。`channel` 配置控制带宽和 RTT，`run.fallback` 选择 A/B，`run.pipeline` 控制完整 Q 的乐观发送。实验脚本的 `--steps` 是草稿块数。相同配置和显式固定 θ 可复现 token 序列；时间包含 CPU 实测项，按时间优化的 θ 可能随机器负载改变。

## 算法与验收口径

- **四种解码。** B0 从目标分布 P 采样。B1 从完整 fp16 log-Q 还原分布，标准接受率为 min(1,P/Q)，拒绝时从归一化的正残差 (P−Q)₊ 抽样。边端也从同一份 fp16 log-Q 定义的 Q 采样。B2 每个位置独立抽一个目标 token，与草稿相等就继续，不等就输出该目标 token 并结束此块。
- **压缩包。** `su/serialize.py` 的 SUP2 格式包含所有候选 ID、N×k 的 fp16 草图、N×r 的 fp16 尾部系数、top-token ID 和原始精度概率，以及稀疏基增量。拟合前屏蔽头部。收发两端用同一量化增量更新基。字节由实际序列化字段核算，链路模型另计协议开销。
- **门控与回退。** 块分数取各位置草图偏差的最大值，支持 L1 或余弦。A 超阈值后使用完整 Q 验证；串行模式在判定后补发，流水线模式预发，门控通过也计入已发送的完整字节。B 超阈值后使用 token-only。压缩路径通过门控后丢弃完整 Q，若近似接受判定拒绝，直接从当时的 P 采样。该路径保持输出在目标分布的支撑内，但可能改变输出概率。
- **校准与消融。** `calibration.json` 记录分位数、接受翻转、KL、TV、AUC 及阈值候选的实测接受率、模拟延迟和字节。校准与评测使用不同的固定合成提示，输出文件保存 token ID，可核对。冒烟用回退率和相对 B1 接受率筛候选，再按估计延迟选 θ；无合格候选则记录 `operating_theta=null`，仍用配置分位数完成对照。消融包含门控开关、A 的串行/流水线、三个分位数、九组 r×k。
- **计时。** 分项 CPU 时间加串行链路仿真；固定 batch=1，无 warmup。`tau` 是平均接受草稿数，`tokens_per_step` 包含纠正和奖励 token，主表 `per_token_ms` 是总时长除以总输出 token。`rtt_measured_ms=null` 表示未测物理 RTT。校准、冒烟和带宽扫描的 JSON 见 [CPU 冒烟记录](evidence/cpu-smoke/summary.json)。

`su/engine.py` 是单进程参考解码循环，`su/channel.py` 定义链路时间线，`su/tables.py` 汇总对照与消融。真实双机传输、并行目标前向优化、真实题集质量和论文性能验收均超出这次本地冒烟。草图与 TV 的关系是统计关系；从单个随机草图不能保证恢复真实 TV。

## 实验与已保存结果

| 实验 | 比较内容 |
|---|---|
| 门控校准 | 用相同随机数比较完整和近似 Q 的接受判定，测逐位置翻转 AUC |
| 四路主对照 | 云端单独解码、完整 Q、仅 token、低秩＋草图，比较字节、接受长度和每 token 延迟 |
| 带宽扫描 | 四种方法各跑 1、5、10、50、100 Mbps，检查延迟曲线交点 |
| 门控开关 | 比较始终使用近似分布和按阈值回退 |
| 回退传输方式 | 对完整 Q 回退比较串行补传和乐观预发 |
| 阈值消融 | 比较校准分数的 90%、95%、99% 分位阈值 |
| 压缩参数消融 | 比较秩 2/4/8 与草图维度 64/128/256 的九组组合 |

随仓样例来自 Windows、Python 3.12、PyTorch 2.9.0 CPU 和 Transformers 4.57.1。8 个草稿块上，完整 Q 与压缩方案平均上行分别为 16,490 和 3,082 字节/步。独立合成校准提示上有 32 个位置、2 个翻转，逐位置 ROC-AUC 为 0.0833。完整 Q 的强制回退在串行和流水线两种模式下与 B1 输出一致。原始 [64 项测试日志](evidence/cpu-smoke/unit.log)、[校准记录](evidence/cpu-smoke/calibration.json)、[带宽扫描](evidence/cpu-smoke/sweep.json)、[消融数据](evidence/cpu-smoke/ablation.json) 和 [强制回退 B 记录](evidence/cpu-smoke/forced-b/run.json) 可分别查看。

Linux shell 中可用 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m su.cli all --steps 8 --out runs/reproduce` 运行同一 CPU 流程。当前入口按 CPU 执行；在有 A100 的主机运行此命令也不会自动使用 GPU。
