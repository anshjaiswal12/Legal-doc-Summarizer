import json
import urllib.request
import os
from datasets import load_dataset

def load_billsum():
    return load_dataset("FiscalNote/billsum", cache_dir="data/raw/billsum/")

def download_cuad_json(target_path: str = "data/raw/cuad/CUADv1.json"):
    import zipfile
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):
        zip_url = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
        zip_path = os.path.join(os.path.dirname(target_path), "data.zip")
        req = urllib.request.Request(zip_url)
        req.add_header("User-Agent", "Mozilla/5.0")
        resp = urllib.request.urlopen(req, timeout=120)
        with open(zip_path, "wb") as f:
            f.write(resp.read())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("CUADv1.json", os.path.dirname(target_path))
    return target_path

def load_cuad_raw(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_cuad(raw: dict):
    flat_rows = []
    for entry in raw["data"]:
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answers = qa["answers"]
                answer_texts = [a["text"] for a in answers]
                answer_starts = [a["answer_start"] for a in answers]
                flat_rows.append({
                    "contract_title": title,
                    "context": context,
                    "question": question,
                    "answer_texts": answer_texts,
                    "answer_starts": answer_starts
                })
    return flat_rows

def group_cuad_by_contract(flat_rows: list):
    grouped = {}
    for row in flat_rows:
        title = row["contract_title"]
        if title not in grouped:
            grouped[title] = []
        grouped[title].append(row)
    return grouped

if __name__ == "__main__":
    print("Loading BillSum...")
    billsum = load_billsum()
    print(f"BillSum splits: {billsum}")
    print(f"Example 0 (Train):")
    example = billsum['train'][0]
    print(f"Title: {example['title']}\nText snippet: {example['text'][:200]}...\nSummary snippet: {example['summary'][:200]}...\n")
    
    print("Loading CUAD...")
    json_path = download_cuad_json()
    raw = load_cuad_raw(json_path)
    flat = flatten_cuad(raw)
    grouped = group_cuad_by_contract(flat)
    
    print(f"CUAD Total flat rows: {len(flat)}")
    print(f"CUAD Total distinct contracts: {len(grouped)}")
    
    first_contract = list(grouped.keys())[0]
    print(f"\nQuestion list for contract '{first_contract}':")
    for row in grouped[first_contract][:5]:
        print(f"- {row['question']}")
        
    assert len(billsum['train']) > 0
    assert len(grouped) > 0
    for contract, rows in grouped.items():
        assert len(rows) > 1, f"Contract {contract} has only {len(rows)} row(s)"
        
    print("PHASE 1 VERIFIED")
