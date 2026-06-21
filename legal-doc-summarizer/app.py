import streamlit as st
import sys
import os
import torch
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import nltk

# Ensure NLTK punkt is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.extractive import lexrank_summarize
from src.summarize_pipeline import summarize_contract
from src.clause_tagger import tag_clauses, clause_coverage
from src.data_prep import download_cuad_json, load_cuad_raw, flatten_cuad, group_cuad_by_contract

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

st.title("⚖️ Legal Document Summarizer & Clause Tagger")
st.markdown("Summarize long legal contracts and inspect clause preservation using fine-tuned LoRA adapters (DistilBART, T5-Small) and LexRank.")

# Cache CUAD database load
@st.cache_resource
def load_cuad_db():
    try:
        json_path = download_cuad_json()
        raw = load_cuad_raw(json_path)
        flat = flatten_cuad(raw)
        grouped = group_cuad_by_contract(flat)
        return grouped
    except Exception as e:
        st.warning(f"Could not load CUAD database: {e}")
        return {}

grouped_cuad = load_cuad_db()

# Cache Model Loading
@st.cache_resource
def load_model(peft_model_id: str):
    config = PeftConfig.from_pretrained(peft_model_id)
    base_model_name = config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, peft_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer

# Side-by-side Input Selection
col_input, col_config = st.columns([2, 1])

with col_input:
    st.subheader("1. Provide Contract Text")
    
    # Optional CUAD contract selection helper
    if grouped_cuad:
        cuad_titles = ["-- Upload/Paste Custom Text --"] + list(grouped_cuad.keys())
        selected_cuad_title = st.selectbox("Or choose a pre-loaded CUAD contract:", cuad_titles)
    else:
        selected_cuad_title = "-- Upload/Paste Custom Text --"
        
    pasted_text = st.text_area("Paste contract text here:", height=300)
    uploaded_file = st.file_uploader("Or upload a contract text file (.txt):", type=["txt"])

with col_config:
    st.subheader("2. Model & Options")
    model_choice = st.selectbox(
        "Select Summarization Model:",
        ["Compare All Three", "LexRank (Extractive)", "DistilBART (Abstractive)", "T5-Small (Abstractive)"]
    )
    use_chunking = st.checkbox("Enable Multi-Chunking & Map-Reduce", value=True)
    
    st.info("Note: Running models on long documents will run chunking and map-reduce automatically if length > 1024 tokens.")

# Determine input text
input_text = ""
if selected_cuad_title != "-- Upload/Paste Custom Text --" and grouped_cuad:
    input_text = grouped_cuad[selected_cuad_title][0]["context"]
elif uploaded_file is not None:
    input_text = uploaded_file.read().decode("utf-8")
else:
    input_text = pasted_text

if st.button("Generate Summary", type="primary"):
    if not input_text.strip():
        st.error("Please provide some contract text first!")
    else:
        with st.spinner("Processing..."):
            # 1. Clause Tagging / Lookup
            tagged_clauses = {}
            is_cuad_match = False
            
            # Try to match with CUAD database
            if grouped_cuad:
                for title, rows in grouped_cuad.items():
                    context = rows[0]["context"]
                    # Substring match to allow slight edits
                    if input_text[:500] in context or context[:500] in input_text:
                        tagged_clauses = tag_clauses(rows)
                        is_cuad_match = True
                        st.success(f"Matched with CUAD contract: **{title}** (Using ground-truth annotations)")
                        break
            
            # Fallback regex tagger for custom uploaded contracts
            if not is_cuad_match:
                st.info("Custom contract: running regex-based clause category tagger...")
                fallback_rules = {
                    "Confidentiality": r"(?i)\b(confidential|non-disclosure|ndas?|proprietary information)\b",
                    "Governing Law": r"(?i)\b(governing law|jurisdiction|venue|applicable law)\b",
                    "Termination": r"(?i)\b(terminate|termination|expir|cancel|default)\b",
                    "Indemnification": r"(?i)\b(indemn|hold harmless|defense)\b",
                    "Intellectual Property": r"(?i)\b(intellectual property|patents?|trademarks?|copyrights?|inventions?)\b"
                }
                sentences = nltk.sent_tokenize(input_text)
                for cat, regex in fallback_rules.items():
                    matched_clauses = []
                    for s in sentences:
                        if re.search(regex, s):
                            matched_clauses.append({"text": s, "start": input_text.find(s)})
                    if matched_clauses:
                        tagged_clauses[cat] = matched_clauses[:3]  # Limit to top 3 matches
            
            # 2. Run Summarizer(s)
            summaries = {}
            
            # LexRank
            if model_choice in ["Compare All Three", "LexRank (Extractive)"]:
                summaries["LexRank (Extractive)"] = lexrank_summarize(input_text, sentence_count=5)
                
            # DistilBART
            if model_choice in ["Compare All Three", "DistilBART (Abstractive)"]:
                try:
                    model_bart, tok_bart = load_model("outputs/models/distilbart-lora")
                    summaries["DistilBART (Abstractive)"] = summarize_contract(
                        input_text, model_bart, tok_bart, use_chunking=use_chunking, prefix=""
                    )
                except Exception as e:
                    st.error(f"Error running DistilBART: {e}")
                    
            # T5-Small
            if model_choice in ["Compare All Three", "T5-Small (Abstractive)"]:
                try:
                    model_t5, tok_t5 = load_model("outputs/models/t5-lora")
                    summaries["T5-Small (Abstractive)"] = summarize_contract(
                        input_text, model_t5, tok_t5, use_chunking=use_chunking, prefix="summarize: "
                    )
                except Exception as e:
                    st.error(f"Error running T5-Small: {e}")
            
            # 3. Display Summaries
            st.write("---")
            st.subheader("📝 Generated Summaries")
            
            if len(summaries) > 1:
                cols = st.columns(len(summaries))
                for idx, (name, summary) in enumerate(summaries.items()):
                    with cols[idx]:
                        st.markdown(f"### {name}")
                        st.write(summary)
            else:
                for name, summary in summaries.items():
                    st.markdown(f"### {name}")
                    st.write(summary)
            
            # 4. Display Clause Tagging Results
            st.write("---")
            st.subheader("🏷️ Clause Preservation Metrics")
            
            if tagged_clauses:
                st.markdown(f"Found **{len(tagged_clauses)}** distinct clause categories in the contract.")
                
                # Show preservation table
                preservation_records = []
                for name, summary in summaries.items():
                    cov = clause_coverage(summary, tagged_clauses)
                    preserved_count = sum(cov.values())
                    rate = preserved_count / len(tagged_clauses) if tagged_clauses else 0.0
                    preservation_records.append({
                        "Model": name,
                        "Preserved Clauses": f"{preserved_count} / {len(tagged_clauses)}",
                        "Preservation Rate": f"{rate:.1%}"
                    })
                st.table(preservation_records)
                
                # Expandable Clause Details
                with st.expander("Show Tagged Clauses"):
                    for cat, clauses in tagged_clauses.items():
                        st.markdown(f"**Category: {cat}**")
                        for c in clauses[:2]:  # Show up to 2 snippets
                            st.caption(f"📍 Start char {c['start']}:")
                            st.info(c['text'])
            else:
                st.warning("No clauses tagged. Make sure the contract text contains key legal terms.")
