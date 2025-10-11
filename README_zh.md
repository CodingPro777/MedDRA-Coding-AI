# MedDRA-Coding-AI（中文介绍）

中文 | [English](README.md)

<p align="center">
  <img src="images/Screenshot 2025-10-10 at 11.25.04 PM.png" alt="对话界面" width="280" />
  <img src="images/Screenshot 2025-10-10 at 11.27.31 PM.png" alt="向量召回" width="280" />
  <img src="images/Screenshot 2025-10-10 at 11.30.01 PM.png" alt="索引构建" width="280" />
</p>

MedDRA-Coding-AI 是一款面向药物警戒（Pharmacovigilance）的开源自动编码工具，利用检索增强生成（Retrieval-Augmented Generation, RAG）技术，将安全性叙述快速映射到 MedDRA 术语。系统支持本地/云端大模型（OpenAI、OpenRouter、Ollama），并提供 Streamlit Web 界面、CLI 工具及 Colab Notebook，适合药物不良事件分类与自动化编码场景。

## 功能亮点
- 解析 MedDRA ASCII 数据，构建完整的层级数据框架。
- 将 LLT 扩展为包含 PT → HLT → HLGT → SOC 全路径的层级文档，再做句向量编码，增强召回质量。
- 向量检索支持 FAISS（默认）与 Chroma（持久化存储），内置 BAAI/bge 系列句向量模型，可按需切换。
- RAG 代理会将检索候选及层级上下文交给 LLM，生成编码建议与推理说明。
- Streamlit 界面支持交互式对话、候选可视化、低置信度提示。
- CLI 脚本 `build_index.py` 可批量扫描版本目录、一次性生成所有向量索引。
- `notebooks/build_meddra_index_colab.ipynb` 允许在 Google Colab 构建大规模索引，缓解本地资源压力。

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r meddra_rag_assistant/requirements.txt
   ```
2. 将 MedDRA ASCII 数据放入 `dict/Meddra/<language_version>/`（例如 `dict/Meddra/english_24.0/`）。
3. 运行索引构建（以 `english_24.0` 为例）：
   ```bash
   python meddra_rag_assistant/build_index.py --versions english_24.0 --force
   ```
4. 启动 UI：
   ```bash
   streamlit run meddra_rag_assistant/main.py
   ```

## 配置说明
编辑 `meddra_rag_assistant/config.yaml` 可调整：
- `embedding`：句向量模型名称、批大小、是否归一化以及设备（`auto`、`cpu`、`cuda`）。
- `vector_store`：`faiss` 或 `chroma`，Chroma 支持自定义 `collection_prefix` 与 `add_batch_size`。
- `retrieval`：默认检索数量、分数阈值。
- `llm`：接入 OpenAI / OpenRouter / Ollama 的模型名称、温度等参数。

环境变量示例：
```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...
```

## Roadmap（规划）
- 支持 WHODrug 药品词典解析与编码，与 MedDRA 结果联动。
- 训练面向不良事件叙述的领域专用句向量模型，提升复杂描述的匹配能力。
- 构建自动化评测基准（召回率、Top-K 精度、人工复核流程）。

欢迎贡献代码或提交 Issue，共同完善药物警戒自动编码生态！
