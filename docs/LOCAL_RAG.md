# 自建 RAG：为什么做、包含什么、与论文有什么不同

本地RAG与原版Agent源码分开上传；此次没有将正在审批的Task1修复、通用核查、强制最终诊疗、其他方法或全量实验控制器加入原版目录。模块代码是已有本地实现的整理，不代表已修复全部端到端流程。

## MedChain：重建病例库＋BGE-M3

原版任务依赖`MDDB.xlsx`、`MDDB_all.xlsx`和症状向量；无法由公开代码确认作者原训练名单与全部建库产物。直接把发布完整数据作为RAG库会使测试病例进入检索。

因此自建方案先从公开MedChain完整病例中，排除测试源ID、相同临床文本及近重复内容，再建立冻结病例库。源码位于：

- `medagent/benchmarks/case_rag.py`：去重与构建、按科室组织、原接口兼容工作簿、BM25检索。BM25为已有本地辅助实现，不是论文主检索器。
- `dense_case_rag.py`：BGE向量的余弦Top-K、查询缓存、向量/ID/哈希一致性校验。
- `paper_memory.py`：优先已验证病例，不足时使用低优先级伪病例；只接受外部显式加载的既有伪病例，不凭空创造病例。
- `scripts/prepare_medchain_rag.py`、`prepare_medchain_dense.py`：构建脚本。

原论文编码器是`text-embedding-ada-002`，这里替换为本地`BAAI/bge-m3`。保持病例单位、科室过滤、余弦Top-3的机制，但排序不会与原模型等价。

默认稠密建库脚本按原版`main.py`使用主诉；`--symptom-summaries`可提供冻结且完整覆盖历史库的≤70字摘要。没有这个参数时，不能把该索引称为已经执行了论文的症状提炼。后续Task1的输入/摘要协议仍需用户审批，本次上传不替用户选择。

## ClinicalAgents：自建指南库＋历史检查证据

官方快照没有`experience_memory.py`，也未附完整指南库及可直接恢复的AGRAG检索实现。因此保留官方源码不改，另外提供：

- `rag/clinicalagents/prepare.py`：固定8个CDC来源的可选抓取、正文提取、分节分段、来源与时间及哈希记录，指南和历史病例向量建库。
- `rag/clinicalagents/experience_memory.py`：冻结索引加载、状态白名单查询、指南检索、历史病例检查类别检索、潜在缺失检查提示。

两个文件分别来自本地`research/clinicalagents_local_complete/prepare.py`和`runtime/experience_memory.py`，仅改变存放位置，文件内容不改。不上传该目录中的主控、Agent修改和收尾阶段。

已有本地快照为8份来源、948个英文段落、9511个清洗后历史病例；这是2026-10-28的构建结果，**网页会变化，重新抓取不能保证相同段落数和内容哈希**。所有指南正文、历史病例和索引均不随本仓库上传。指南偏感染、结核及疼痛，不能称全科覆盖。

该方案使用BGE-M3、CLS+L2归一化、1024维。指南/历史检索阈值0.35、潜在缺失检查阈值0.25，是未校准开发参数；检索是静态向量方案，**不是论文原版AGRAG或完整实体/图谱检索**。历史病例不编码诊断和治疗标签，不作为当前患者已发生的发现。查询允许当前模型的疾病假设，不使用测试参考诊断/治疗。检索结果标记外部知识来源。

## BGE编码器准备

BGE是检索编码器，不是LLM。仅在使用自建RAG时需要；不会在MedChain数据下载中顺带下载。

```bash
# 在已有torch/transformers环境运行，仓库根目录执行
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
path = Path('data/models/bge-m3')
revision = '5617a9f61b028005a4858fdac845db406aefb181'
snapshot_download('BAAI/bge-m3', revision=revision, local_dir=path)
(path / 'download_manifest.json').write_text(json.dumps({
    'model': 'BAAI/bge-m3', 'revision': revision
}, indent=2))
PY
```

所有下列命令会写入被Git忽略的`data/`目录。它们不是此次上传已经执行过的下载或建库。

## 重建MedChain RAG

先按[数据下载说明](DATA_DOWNLOAD.md)准备数据，再执行：

```bash
python scripts/prepare_medchain_rag.py \
  --output-dir data/benchmarks/medchain-rag-bm25-v1-20261028
python scripts/prepare_medchain_dense.py \
  --corpus-dir data/benchmarks/medchain-rag-bm25-v1-20261028 \
  --output-dir data/benchmarks/medchain-dense-local \
  --model-dir data/models/bge-m3 --device cpu
```

使用示例：

```python
from medagent.benchmarks.dense_case_rag import DenseCaseRetriever
retriever = DenseCaseRetriever(
    'data/benchmarks/medchain-rag-bm25-v1-20261028',
    'data/benchmarks/medchain-dense-local')
# 默认索引只接受构建时缓存过的主诉；新查询必须编码后重建或使用PaperMemory。
hits = retriever.search(query='构建索引时已缓存的患者主诉', department=None, k=3)
```

`PaperMemory`的未缓存查询调用本地23830端口；启动编码服务：

```bash
python scripts/serve_case_embeddings.py --model-dir data/models/bge-m3 --port 23830
```

这里只提供检索层，不自动重写原`main.py`的RAG调用，也不冒充已经接回原始MDDB向量接口。`PaperMemory`需要的原wave调度、伪病例写入属于流程编排，未把实验控制器一起上传。

## 可选：重建ClinicalAgents指南RAG

此步骤会另行访问CDC网站，仅需指南RAG时手动执行，不属于MedChain benchmark数据下载。

```bash
python rag/clinicalagents/prepare.py fetch \
  --guidance data/benchmarks/clinicalagents-guidelines-local
# 原有index函数使用CUDA；必须选择空闲卡。历史库路径就是上面的固定路径。
CUDA_VISIBLE_DEVICES=0 python rag/clinicalagents/prepare.py index \
  --guidance data/benchmarks/clinicalagents-guidelines-local \
  --index data/benchmarks/clinicalagents-bge-local
python scripts/serve_case_embeddings.py --model-dir data/models/bge-m3 --port 23831
```

程序化调用：

```python
from rag.clinicalagents.experience_memory import ExperienceMemory
memory = ExperienceMemory(
    'data/benchmarks/clinicalagents-bge-local/guidelines.json',
    'data/benchmarks/clinicalagents-bge-local/cases.json',
    held_out_case_ids=[]  # 正式使用时填完整测试ID，用于交叉检查
).freeze()
result = memory.retrieve({'chief_complaint': '已知主诉'})
```

模块默认查询23831端口，可由`CA_EMBED_PORT`指定。构建、查询编码器版本必须一致。正式使用要传完整测试病例ID；上例只展示接口，不表示省略检查是实验协议。

官方ClinicalAgents绝对导入`experience_memory`，不会自动发现这个独立包；要接入需单独配置桥接与缺失的`app_config.py`。专业Agent/rollout同模型限制也仍存在。本次没有借上传之机修改这些原始约束。

## 上传边界

上传检索代码、固定来源URL、构建逻辑与说明；不上传指南网页/正文、病例JSON、工作簿、向量、权重、测试输出。源码快照与自建部分通过`UPSTREAM_SOURCES.json`和`UPLOAD_MANIFEST.json`分别追踪。
