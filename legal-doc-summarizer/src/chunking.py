import nltk
from nltk.tokenize import sent_tokenize

def chunk_document(text: str, tokenizer, max_tokens: int = 1024, overlap: int = 100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    i = 0
    while i < len(sentences):
        sentence_tokens = tokenizer.encode(sentences[i], add_special_tokens=False)
        if len(sentence_tokens) > max_tokens:
            split_idx = 0
            while split_idx < len(sentence_tokens):
                chunk_slice = sentence_tokens[split_idx : split_idx + max_tokens]
                chunks.append(tokenizer.decode(chunk_slice, skip_special_tokens=True))
                split_idx += max_tokens - overlap
            i += 1
            continue
            
        if current_length + len(sentence_tokens) <= max_tokens:
            current_chunk.append(sentences[i])
            current_length += len(sentence_tokens)
            i += 1
        else:
            chunks.append(" ".join(current_chunk))
            overlap_length = 0
            overlap_sentences = []
            for j in range(len(current_chunk)-1, -1, -1):
                stoks = tokenizer.encode(current_chunk[j], add_special_tokens=False)
                if overlap_length + len(stoks) > overlap and overlap_sentences:
                    break
                overlap_sentences.insert(0, current_chunk[j])
                overlap_length += len(stoks)
            current_chunk = overlap_sentences
            current_length = overlap_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def reduce_summaries(chunk_summaries: list) -> str:
    return " ".join(chunk_summaries)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_prep import load_billsum
    
    billsum = load_billsum()
    # Find a long document
    long_doc = None
    for item in billsum['train']:
        if len(item['text']) > 5000:
            long_doc = item['text']
            break
            
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    chunks = chunk_document(long_doc, tokenizer, max_tokens=1024, overlap=100)
    
    print(f"Produced {len(chunks)} chunks.")
    for idx, c in enumerate(chunks):
        toks = len(tokenizer.encode(c, add_special_tokens=False))
        print(f"Chunk {idx} token count: {toks}")
        assert toks <= 1024, f"Chunk {idx} exceeds max_tokens"
        
    print("PHASE 2 VERIFIED")
