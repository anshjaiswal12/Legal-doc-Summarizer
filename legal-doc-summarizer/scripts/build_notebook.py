import nbformat as nbf
import os

def build():
    os.makedirs("notebooks", exist_ok=True)
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell("## 1. Data Loading"),
        nbf.v4.new_code_cell("""import sys, os
sys.path.append(os.path.abspath('..'))
from src.data_prep import load_billsum, download_cuad_json, load_cuad_raw, flatten_cuad, group_cuad_by_contract
billsum = load_billsum()
print("BillSum sizes:", {k: len(v) for k, v in billsum.items()})

json_path = download_cuad_json('../data/raw/cuad/CUAD_v1.json')
raw = load_cuad_raw(json_path)
flat = flatten_cuad(raw)
grouped = group_cuad_by_contract(flat)
print("CUAD Total flat rows:", len(flat))
print("CUAD Total distinct contracts:", len(grouped))"""),

        nbf.v4.new_markdown_cell("## 2. Preprocessing — Chunking & Tokenization"),
        nbf.v4.new_code_cell("""from src.chunking import chunk_document
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-small')
text = billsum['train'][0]['text']
chunks = chunk_document(text, tokenizer, max_tokens=1024, overlap=100)
print(f"Produced {len(chunks)} chunks for the contract.")"""),

        nbf.v4.new_markdown_cell("## 3. Extractive Baseline — LexRank"),
        nbf.v4.new_code_cell("""from src.extractive import lexrank_summarize
for i in range(3):
    text = billsum['train'][i]['text']
    summary = lexrank_summarize(text, sentence_count=3)
    print(f"Summary {i+1}: {summary}\\n")"""),

        nbf.v4.new_markdown_cell("## 4. Fine-Tuning — DistilBART (Abstractive)\\nLoading the saved adapter to save time instead of re-training."),
        nbf.v4.new_code_cell("""from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

peft_model_id = '../outputs/models/distilbart-lora'
try:
    config = PeftConfig.from_pretrained(peft_model_id)
    base_model_name = config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, peft_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    text = billsum['train'][500]['text']
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=300)
    print("DistilBART Summary:", tokenizer.decode(out[0], skip_special_tokens=True))
except Exception as e:
    print(f"Model load skipped: {e}")"""),

        nbf.v4.new_markdown_cell("## 5. Fine-Tuning — T5-Small (Comparison)\\nLoading the saved adapter."),
        nbf.v4.new_code_cell("""peft_model_id = '../outputs/models/t5-lora'
try:
    config = PeftConfig.from_pretrained(peft_model_id)
    base_model_name = config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, peft_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    text = billsum['train'][500]['text']
    inputs = tokenizer("summarize: " + text, max_length=1024, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=300)
    print("T5-Small Summary:", tokenizer.decode(out[0], skip_special_tokens=True))
except Exception as e:
    print(f"Model load skipped: {e}")"""),

        nbf.v4.new_markdown_cell("## 6. Clause-Type Tagging (CUAD)"),
        nbf.v4.new_code_cell("""from src.clause_tagger import tag_clauses
test_contract = list(grouped.keys())[0]
tagged = tag_clauses(grouped[test_contract])
print(f"Contract: {test_contract}")
for cat, clauses in tagged.items():
    print(f" - {cat}: {len(clauses)} clauses found")"""),

        nbf.v4.new_markdown_cell("## 7. Evaluation — ROUGE & Clause Preservation"),
        nbf.v4.new_code_cell("""import pandas as pd
from src.evaluate import compare_models
import json
with open('../outputs/summaries/ten_contract_results.json', 'r') as f:
    res = json.load(f)
df = pd.DataFrame(res)
print(df[['contract_id', 'original_length', 'clause_types_found', 'extractive_clause_types_preserved', 'abstractive_clause_types_preserved']].head())"""),

        nbf.v4.new_markdown_cell("## 8. Ten-Contract Summarization Run"),
        nbf.v4.new_code_cell("""print("Ten Contract Results loaded from Phase 8:")
print(df.to_string())""")
    ]

    with open('notebooks/main.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Notebook notebooks/main.ipynb generated.")

if __name__ == "__main__":
    build()
