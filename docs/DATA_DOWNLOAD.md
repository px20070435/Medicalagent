# 仅下载 MedChain 数据

本仓库当前树不包含任何数据集。脚本只下载 MedChain，不下载 MedR-Bench 或其他 benchmark，也不下载任何模型。

来源：

- 数据：https://huggingface.co/datasets/ljwztc/MedChain ，固定 revision `769a6de3333bf1ab5e283d2ccbdba62029686a07`。
- 发布测试标注和科室表：https://github.com/ljwztc/MedChain/tree/35ab53e3c601831a7bca181c72ecad6b12b55ff3/datasets 。下载后的 SHA-256 必须匹配 `configs/medchain-download.json`。

在仓库根目录执行（Python 3.10+）：

```bash
python -m pip install -r requirements-tools.txt
python scripts/download_medchain.py --plan  # 只显示来源，不下载
python scripts/download_medchain.py
```

图像压缩包较大，磁盘需同时容纳压缩包和解压内容。Hugging Face 下载可以利用缓存继续；脚本拒绝覆盖内容不同的既有标注。

生成目录：

```text
data/raw/medchain/                 # merged_cases.json、报告及 images.zip
data/image_archive/                # 解压图像
data/medchain-download-manifest.json
datasets/filtered_data_test_set.json
datasets/merged_cases_room_structure.txt
datasets/merged_cases.json         # 指向本地下载内容
datasets/image_reports.json
datasets/images_with_reports.txt
datasets/MedImg/                   # 平铺图像软链接
```

同名不同内容图像会明确报错，避免静默覆盖。标注和原始完整病例不应混用为训练/测试集：自建病例库必须排除测试病例及重复内容，见RAG说明。

下载数据不会生成作者未公开的原始训练名单、MDDB向量或ClinicalAgents指南库，也不表示获得了完全相同的论文划分。CDC指南下载是可选RAG步骤，和这里的MedChain benchmark下载分开执行。
