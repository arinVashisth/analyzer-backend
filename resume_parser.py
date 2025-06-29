import pdfplumber

def extract_resume_text(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1)
            if page_text:
                text += page_text + '\n'
    return text.strip()
