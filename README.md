# Medicalagent：原始基线与本地 RAG

仅保留原版 MedChain、ClinicalAgents 官方源码快照，以及相关数据下载、InternVL2-8B 服务和自建 RAG 工具。原始算法源码与本地检索实现分开，未上传待审批的 Task1 修复或其他研究方法。

| 路径 | 内容 |
|---|---|
| `main.py`、`task_framework/`、`utils/`、`my_prompt/`、`doctor_patient_interaction/` | MedChain 固定版本原始源码、相关提示词和医患交互代码 |
| `research_methods/clinicalagents/upstream/` | ClinicalAgents 固定版本官方源码和提示词，未改写 |
| `medagent/benchmarks/` | 自建 MedChain 病例库去重、BM25/稠密检索、伪病例记忆；辅助加载契约在 `medagent/core/` |
| `rag/clinicalagents/` | 自建 CDC 指南与历史病例检索、建库脚本；不包含本地主控和强制最终诊疗变体 |
| `scripts/` | 本仓库所需下载、建库、检索编码和 InternVL 服务脚本 |

- [MedChain 数据下载](docs/DATA_DOWNLOAD.md)
- [InternVL2-8B 配置](docs/INTERNVL_CONFIG.md)
- [自建 RAG 的原因、差异与使用](docs/LOCAL_RAG.md)
- [上传范围及逐文件清单](UPLOAD_CONTENTS.md)
- [源码版本记录](UPSTREAM_SOURCES.json)

`datasets/`、`data/`、模型权重、病例内容、指南正文、索引、实验输出均不在当前版本中。先阅读下载说明，再在本地生成。

这是一份源码与检索工具整理，**不是宣称 clone 后即可完整复现实验**：MedChain 原始代码依赖未随源码提供的 MDDB/embedding 配置，存在原有缺失模板；ClinicalAgents 快照缺少 `app_config.py` 与 `experience_memory.py`，并要求 rollout 模型不同于专业 Agent。详见各说明。未为方便运行而静默修改原始算法。

MedChain 来源：https://github.com/ljwztc/MedChain ，commit `35ab53e3c601831a7bca181c72ecad6b12b55ff3`。
ClinicalAgents 来源：https://github.com/ZhuohanGe/ClinicalAgents-Code ，commit `e7dbd15513235e388cbb0dc0afafdfcbacefe420`。
ClinicalAgents 快照没有附带 LICENSE；根目录原有 MedChain MIT LICENSE 不替代第三方代码自身的授权说明。
