import zipfile
import os

def bundle():
    with zipfile.ZipFile('legal-doc-summarizer-submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('notebooks/main.ipynb')
        zipf.write('outputs/reports/five_contract_summaries.pdf')
        zipf.write('outputs/reports/extractive_vs_abstractive_report.docx')
        zipf.write('outputs/summaries/ten_contract_results.json')
        for root, _, files in os.walk('src'):
            for f in files:
                zipf.write(os.path.join(root, f))
    print("Zip created: legal-doc-summarizer-submission.zip")
    
    with zipfile.ZipFile('legal-doc-summarizer-submission.zip', 'r') as zipf:
        print(zipf.namelist())
        
    print("PHASE 10 VERIFIED — SUBMISSION READY")

if __name__ == "__main__":
    bundle()
