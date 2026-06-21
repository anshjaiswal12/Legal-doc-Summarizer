import json

def summarize_contract(text: str, model, tokenizer, use_chunking: bool = True, prefix: str = "") -> str:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.chunking import chunk_document, reduce_summaries
    import torch
    
    device = model.device
    
    if use_chunking and len(tokenizer.encode(text, add_special_tokens=False)) > 1024:
        chunks = chunk_document(text, tokenizer, max_tokens=1024, overlap=100)
        chunk_summaries = []
        for c in chunks:
            inputs = tokenizer(prefix + c, max_length=1024, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_length=200)
            chunk_summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))
        
        combined = reduce_summaries(chunk_summaries)
        inputs_final = tokenizer(prefix + combined, max_length=1024, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out_final = model.generate(**inputs_final, max_length=300)
        return tokenizer.decode(out_final[0], skip_special_tokens=True)
    else:
        inputs = tokenizer(prefix + text, max_length=1024, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=300)
        return tokenizer.decode(out[0], skip_special_tokens=True)

def run_full_pipeline(contract_texts: list, cuad_lookup: dict, abstractive_model, tokenizer, prefix=""):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.extractive import lexrank_summarize
    from src.clause_tagger import clause_coverage, tag_clauses
    from src.evaluate import compute_rouge
    
    results = []
    
    for idx, text in enumerate(contract_texts):
        contract_title = list(cuad_lookup.keys())[idx] if idx < len(cuad_lookup) else None
        tagged_clauses = {}
        if contract_title:
            tagged_clauses = tag_clauses(cuad_lookup[contract_title])
        
        ext_sum = lexrank_summarize(text, 5)
        abs_sum = summarize_contract(text, abstractive_model, tokenizer, use_chunking=True, prefix=prefix)
        
        ext_cov = clause_coverage(ext_sum, tagged_clauses)
        abs_cov = clause_coverage(abs_sum, tagged_clauses)
        
        results.append({
            "original_length": len(text),
            "contract_id": contract_title,
            "extractive_summary": ext_sum,
            "abstractive_summary": abs_sum,
            "extractive_clause_coverage": ext_cov,
            "abstractive_clause_coverage": abs_cov,
            "clause_types_found": len(tagged_clauses),
            "extractive_clause_types_preserved": sum(ext_cov.values()),
            "abstractive_clause_types_preserved": sum(abs_cov.values())
        })
        
    return results

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_prep import download_cuad_json, load_cuad_raw, flatten_cuad, group_cuad_by_contract
    from src.clause_tagger import tag_clauses
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    
    json_path = download_cuad_json()
    raw = load_cuad_raw(json_path)
    flat = flatten_cuad(raw)
    grouped = group_cuad_by_contract(flat)
    
    selected_contracts = {}
    for title, rows in grouped.items():
        tagged = tag_clauses(rows)
        if len(tagged) >= 5:
            selected_contracts[title] = rows
        if len(selected_contracts) == 10:
            break
            
    contract_texts = []
    for title, rows in selected_contracts.items():
        context = rows[0]["context"]
        for r in rows:
            assert r["context"] == context, "Context mismatch inside the same contract!"
        contract_texts.append(context)
        
    try:
        peft_model_id = "outputs/models/distilbart-lora"
        config = PeftConfig.from_pretrained(peft_model_id)
        base_model_name = config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(model, peft_model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Skipping abstractive pipeline due to model load error: {e}")
        model, tokenizer = None, None
        
    if model:
        results = run_full_pipeline(contract_texts, selected_contracts, model, tokenizer, prefix="")
        
        os.makedirs("outputs/summaries", exist_ok=True)
        with open("outputs/summaries/ten_contract_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        print("\nPipeline Results (10 contracts):")
        print(f"{'Contract':<30} | {'Len':<8} | {'Clauses':<8} | {'Ext Pres':<8} | {'Abs Pres':<8}")
        for r in results:
            print(f"{str(r['contract_id'])[:30]:<30} | {r['original_length']:<8} | {r['clause_types_found']:<8} | {r['extractive_clause_types_preserved']:<8} | {r['abstractive_clause_types_preserved']:<8}")
            
        assert len(results) == 10
        for r in results:
            assert r["extractive_summary"]
            assert r["abstractive_summary"]
        
        print("PHASE 8 VERIFIED")
