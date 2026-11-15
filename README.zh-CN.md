# CT-WM

**面向O-RAN的校准世界模型、预测诊断与短视界离线规划。**

[English](README.md) · [实验命令](docs/EXPERIMENTS.md) ·
[结果与边界](docs/RESULTS.md)

CT-WM将概率KPI预测、区间校准与低开销候选动作筛选组织为统一的O-RAN决策支持框架。
多尺度状态空间预测器提供未来KPI分布，轻量MPPI比较候选动作，外置安全门控用于
PRB/MCS联合控制实验。Python评估环路支持KPM/RC语义交互、OLLA及冻结SAC/PPO策略。

## CT-WM的设计重点

- **预测校准：** 在验证集拟合区间缩放系数，分别评价覆盖率、区间宽度与动作选择。
- **预算感知规划：** 以冻结预测权重执行低采样数的8步MPPI，并测量实际决策耗时。
- **风险评估：** 联合PRB/MCS控制与外置回退结合，通过长周期、模块消融和多SNR对照检验适用边界。

上述重点属于框架与评估设计；预测主干及基础训练目标复用已有研究，具体来源见
[代码来源与署名](third_party/README.md)。
CT-WM中的“Trusted”表示评估目标，不表示已获得在线安全认证。

## 直接运行

复用已有Python/PyTorch/CUDA环境；不需要新建虚拟环境。先检查，仅补缺失包：

```bash
python scripts/check_environment.py --rl
python -m pip install -e . --no-deps --no-build-isolation
python scripts/demo.py --steps 16
python -m unittest discover -s tests -v
```

CPU示例用明确标记的合成输入演示KPM—OLLA—RC交互，不作为性能评测数据。
图捕获规划器需要CUDA。数据、权重和正式评测流程为：

```bash
# 自行从IEEE DataPort授权下载10个CSV，分别放入data/raw/
python scripts/fetch_weights.py
python scripts/prepare_data.py --window 32 --check-hashes
python scripts/smoke_gpu.py
python experiments/e2_experiment.py verify
```

原始CSV不合并、不上传。训练/验证/测试按文件7/1/2划分，滑窗不跨文件，
标准化仅拟合训练集。Release提供冻结世界模型与此前SAC/PPO权重，正式评测
直接加载；下载器校验SHA-256，不自动重新训练。

## 主要结果

| 任务 | 已保存结果 |
|---|---|
| 一步RSRP预测 | MAE 0.135317 dB，RMSE 0.264605 dB，11,867个共同测试标签 |
| 离线区间诊断 | 50%/95%区间覆盖率51.344%/95.601% |
| 8步模型内候选筛选 | PRB节省17.851%，谱效变化+1.577% |
| 8步决策计算 | 平均12.284 ms，P95 14.945 ms，RTX 3090热启动 |

上述规划收益来自同一冻结模型内部，真实网络反事实收益尚未独立验证。
128步联合控制属于另一Python链路环境，其谱效变化+0.478%、PRB节省−0.572%，
不能与8步收益混用。持值预测匹配对照没有显示学习预测的额外控制收益；
10 dB与15 dB存在明显尾部退化，完整结果保留于结果说明。

## 开源内容与许可

包含可运行Python代码、实验配置、冻结权重下载、汇总结果、中英文使用说明与测试。未包含原始第三方数据、审稿记录、
C++/ns-3代码或Docker RIC。KPM/RC对象是本地JSON语义模拟，不是ASN.1 E2协议栈。

代码采用[MIT许可](LICENSE)，上游许可见[NOTICE](NOTICE)及
[third_party](third_party/README.md)。原始数据必须遵守IEEE DataPort条目条款。
本项目引用格式见[CITATION.cff](CITATION.cff)；使用预测主干时，请同时引用
[署名文档](third_party/README.md)列出的原始研究。
