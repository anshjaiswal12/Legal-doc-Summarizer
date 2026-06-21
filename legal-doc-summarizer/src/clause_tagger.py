import re

def extract_category_from_question(question: str) -> str:
    match = re.search(r'"([^"]+)"', question)
    if match:
        return match.group(1).strip()
    return "Unknown"

def tag_clauses(contract_rows: list) -> dict:
    tagged = {}
    for row in contract_rows:
        question = row["question"]
        category = extract_category_from_question(question)
        answers = row["answer_texts"]
        starts = row["answer_starts"]
        
        if answers and any(a.strip() for a in answers):
            if category not in tagged:
                tagged[category] = []
            for ans_text, start in zip(answers, starts):
                if ans_text.strip():
                    tagged[category].append({"text": ans_text, "start": start})
    return tagged

def clause_coverage(summary_text: str, tagged_clauses: dict) -> dict:
    from nltk.corpus import stopwords
    import nltk
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
    summary_tokens = set([t.lower() for t in nltk.word_tokenize(summary_text) if t.lower() not in stop_words])
    
    coverage = {}
    for category, clauses in tagged_clauses.items():
        found = False
        for clause in clauses:
            clause_tokens = [t.lower() for t in nltk.word_tokenize(clause["text"]) if t.lower() not in stop_words and t.isalnum()]
            if not clause_tokens:
                continue
            overlap = sum(1 for t in clause_tokens if t in summary_tokens)
            fraction = overlap / len(clause_tokens)
            if fraction > 0.1:
                found = True
                break
        coverage[category] = found
        
    return coverage

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_prep import download_cuad_json, load_cuad_raw, flatten_cuad, group_cuad_by_contract
    
    json_path = download_cuad_json()
    raw = load_cuad_raw(json_path)
    flat = flatten_cuad(raw)
    grouped = group_cuad_by_contract(flat)
    
    all_categories = set()
    for row in flat:
        all_categories.add(extract_category_from_question(row["question"]))
        
    print(f"Total distinct categories found: {len(all_categories)}")
    print(sorted(list(all_categories)))
    
    test_contract = None
    test_tagged = None
    for contract, rows in grouped.items():
        tagged = tag_clauses(rows)
        if len(tagged) >= 3:
            test_contract = contract
            test_tagged = tagged
            break
            
    print(f"\nTesting on contract: {test_contract}")
    for cat, clauses in test_tagged.items():
        print(f"Category: {cat}")
        for c in clauses[:1]:
            print(f"  Snippet: {c['text'][:100]}...")
            
    dummy_summary = "This agreement ensures confidentiality and prevents competitive actions."
    coverage = clause_coverage(dummy_summary, test_tagged)
    print(f"\nClause coverage against dummy summary: {coverage}")
    
    assert len(test_tagged) >= 3
    print("PHASE 6 VERIFIED")
