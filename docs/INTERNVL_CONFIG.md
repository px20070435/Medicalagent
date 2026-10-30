# 本地 LLM：InternVL2-8B

目前只记录这一种生成/视觉模型，不上传权重，也不加入其他LLM的下载或服务脚本。

- 仓库：`OpenGVLab/InternVL2-8B`
- 固定 revision：`6fb9ad6924f69424e57fab2ab061d707688f0296`
- 约8B参数；本地目录：`data/models/InternVL2-8B`
- 服务：LMDeploy `0.6.2.post1`，BF16，tensor parallel=1，单卡。
- 端口23333，模型ID `InternVL2-8B`，temperature=0。
- 服务默认session_len=8192；需要时显式设置32768并记录。显存由上下文、图像和KV缓存共同决定，不能把此前机器用量当固定规格。

可选的权重下载命令（本次整理不会执行）：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenGVLab/InternVL2-8B', revision='6fb9ad6924f69424e57fab2ab061d707688f0296', local_dir='data/models/InternVL2-8B')"
```

建立单独的服务环境，避免覆盖原版算法依赖：

根目录`requirements.txt`保留上游Conda环境导出，并非标准pip依赖列表；不要将其与本地服务的pip requirements混用。

```bash
conda create -n internvl2 python=3.10 -y
conda activate internvl2
pip install -r requirements-internvl2-backend.txt
CUDA_VISIBLE_DEVICES=0 PORT=23333 bash scripts/serve_internvl2.sh
```

可设置 `MODEL_PATH`、`LMDEPLOY_BIN`、`PORT`、`SESSION_LEN`；服务脚本只加载本地模型，不会自动下载或停止已有GPU作业。启动前自行选择空闲设备。

机器可读记录见 `configs/internvl2.json`。**这个文件是配置说明，不会被原始代码自动读取。**

MedChain原始 `utils/llm_api.py`使用OpenAI兼容接口，端口23333，且从`/v1/models`读取第一个模型ID；原版客户端中base_url及占位key为硬编码。需要变更时，在本地明确配置为`http://127.0.0.1:23333/v1`和占位key `EMPTY`；此类运行配置编辑与上传的原版源码分开。原图像请求传路径，服务是否接受取决于后端；本次没有混入此前base64客户端适配。

ClinicalAgents原快照需要`app_config.py`中的`LLM_CONFIG`、`MCTS_PLANNING_CONFIG`、`MCTS_ROLLOUT_CONFIG`、`JUDGE_CONFIG`。专业/规划/裁判可记录同一InternVL服务，但**未修改源码会拒绝与专业Agent同名的rollout模型**；另缺经验记忆模块，不能声称只填这个模型配置就能直接运行。

InternVL2不提供本项目所用的BGE embedding功能。BGE-M3是RAG检索编码器，不是另一个LLM，其配置和下载准备单独记在RAG说明。
