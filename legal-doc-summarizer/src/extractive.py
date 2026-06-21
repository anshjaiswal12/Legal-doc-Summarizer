from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

def lexrank_summarize(text: str, sentence_count: int = 5) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_prep import load_billsum
    
    billsum = load_billsum()
    doc_text = billsum['train'][0]['text']
    summary = lexrank_summarize(doc_text, sentence_count=5)
    
    print(f"Original length: {len(doc_text)} chars")
    print(f"Summary length: {len(summary)} chars")
    print(f"Generated summary:\n{summary}")
    
    assert len(summary) > 0
    assert len(summary) < len(doc_text)
    print("PHASE 3 VERIFIED")
