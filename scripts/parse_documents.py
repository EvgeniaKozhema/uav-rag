import os
import fitz

from bs4 import BeautifulSoup

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''

    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

        for script in soup(['script', 'style']):
            script.decompose()

        text = soup.get_text(separator='\n')
        return text

def main():
    for filename in os.listdir(RAW_DIR):
        raw_path = os.path.join(RAW_DIR, filename)
        processed_path = os.path.join(PROCESSED_DIR, os.path.splitext(filename)[0]+'.txt')

        if filename.lower().endswith('pdf'):
            text = extract_text_from_pdf(raw_path)
        elif filename.lower().endswith('html') or filename.lower().endswith('htm'):
            text = extract_text_from_html(raw_path)
        else:
            continue

        with open(processed_path, 'w', encoding='utf-8') as f_out:
            f_out.write(text)

if __name__=='__main__':
    main()