# ⚖️ Legal Document Summarizer & Clause Tagger

An interactive, premium web application designed to automatically summarize long-form legal contracts and evaluate key clause preservation. Built using fine-tuned LoRA adapters (DistilBART and T5-Small) along with LexRank extractive baselines, it facilitates comparative analysis of summarization performance and ensures that critical legal constraints are not lost.

## 🚀 Quick Start (Single-Command Setup)

To configure the virtual environment, download all required models/dependencies, and launch the interactive Streamlit application, simply run:

```bash
chmod +x legal-doc-summarizer/setup_env.sh
./legal-doc-summarizer/setup_env.sh
```

Once the execution is complete, the console will output the local URL (typically `http://localhost:8501`) to access the interface.

---

## 📸 Application Screenshot

Here is the dashboard interface demonstrating the side-by-side model comparison and clause preservation tracking:

![Dashboard Mockup](legal-doc-summarizer/screenshots/app_mockup.png)

---

## 🌟 Key Features

1. **Comparative Summarization**: View side-by-side outputs from three models:
   - **LexRank (Extractive)**: Highlight key sentences directly from the source text.
   - **DistilBART (Abstractive)**: Fine-tuned with LoRA on legal subsets for coherent summaries.
   - **T5-Small (Abstractive)**: Fine-tuned with LoRA for concise summaries.
2. **Clause Tagger & Preservation Metrics**: Detects crucial clause categories (e.g., Confidentiality, Governing Law, Termination, Indemnification, Intellectual Property) and measures the preservation rate in each generated summary.
3. **Chunking & Map-Reduce**: Automatically processes large documents (above 1,024 tokens) utilizing an overlapping chunking window to prevent context limits.

---

## 📄 Test Documents

For evaluation, a realistic, multi-clause Master Services Agreement (MSA) is provided in:
* **[sample_contract.txt](legal-doc-summarizer/sample_contract.txt)**

You can copy and paste or upload this text directly into the web application to evaluate its chunking, tagger, and summarization pipelines.
