# Project State

## Directory Layout
```
legal-doc-summarizer/
├── data/raw/
├── data/processed/
├── src/
│   ├── data_prep.py
│   ├── chunking.py
│   ├── extractive.py
│   ├── train_lora.py
│   ├── clause_tagger.py
│   ├── evaluate.py
│   └── summarize_pipeline.py
├── scripts/
│   ├── verify_env.py
│   ├── build_notebook.py
│   └── build_deliverables.py
├── notebooks/
│   ├── main.ipynb
│   └── main_executed.ipynb
├── outputs/summaries/
│   └── ten_contract_results.json
├── outputs/models/
│   ├── distilbart-lora/
│   └── t5-lora/
├── outputs/reports/
│   ├── five_contract_summaries.pdf
│   └── extractive_vs_abstractive_report.docx
├── tests/
├── app.py
├── run_app.sh
├── bundle_submission.py
├── requirements.txt
├── setup_env.sh
└── project_state.md
```

## Completed Components
- **Phase 0:** Project skeleton, virtual environment setup, and environment verification.
- **Phase 1:** Data acquisition & inspection (BillSum + CUAD raw JSON loading).
- **Phase 2:** Text chunking module (fixed overlap infinite loop bug).
- **Phase 3:** Extractive summarization baseline (LexRank via sumy).
- **Phase 4:** LoRA fine-tuning for DistilBART (SSHLifer/distilbart-cnn-12-6) on BillSum subset.
- **Phase 5:** LoRA fine-tuning for T5-Small on BillSum subset.
- **Phase 6:** Clause tagger (CUAD-based category grouping & token-overlap coverage check).
- **Phase 7:** Evaluation harness (custom ROUGE & clause preservation rate calculations).
- **Phase 8:** Full pipeline integration & 10-contract batch summarization execution.
- **Phase 9:** Jupyter notebook assembly and successful end-to-end execution.
- **Phase 10:** Final deliverables packaging (PDF summary report, DOCX comparison report, and submission zip bundle).
- **Streamlit Web Application:** Interactive UI (`app.py` & `run_app.sh`) enabling text copy-paste/upload and comparative summarization.

## Active Variables / Config
- BASE_MODEL_ABSTRACTIVE = "sshleifer/distilbart-cnn-12-6"
- BASE_MODEL_BASELINE = "t5-small"
- TRAIN_DATASET = "FiscalNote/billsum" (HuggingFace Hub)
- CLAUSE_DATASET_SOURCE = "https://github.com/TheAtticusProject/cuad/raw/main/data/CUAD_v1.json"
- DEVICE = "cuda" (RTX 3060 Laptop GPU active)
- MAX_INPUT_TOKENS = 1024
- CHUNK_OVERLAP = 100
- LORA_RANK = 8
- INSTALLED_VERSIONS:
  - torch: 2.12.1+cu130
  - transformers: 5.12.1
  - peft: 0.19.1
  - datasets: 5.0.0
  - streamlit: 1.58.0

## Outstanding Targets
- None (All core phases completed, tested, and fully verified).
