# ClinicalAgents 官方源码快照

`upstream/`保留29个官方文件，commit `e7dbd15513235e388cbb0dc0afafdfcbacefe420`，未改写源码或提示词。来源：https://github.com/ZhuohanGe/ClinicalAgents-Code 。`UPSTREAM_MANIFEST.json`记录原文件哈希。

离线核验（不导入上游，不调用模型）：

```bash
python research_methods/clinicalagents/check_snapshot.py
```

快照完整不代表可直接运行：官方缺少`app_config.py`、`experience_memory.py`，且原主控要求rollout模型不同于专业Agent。所有Agent统一用InternVL会触及这一约束，本次保留原限制，不加入绕过逻辑。

自建检索独立位于`rag/clinicalagents/`；原因、索引重建和接入边界见[本地RAG说明](../../docs/LOCAL_RAG.md)。没有上传此前本地主控和强制最终诊断/治疗变体。

上游快照未附LICENSE；根目录许可证不替代该源码自身授权说明。
