import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def extract_text_from_multiple_pdfs(pdf_paths):
    """Extracts text from multiple PDFs."""
    all_text = {}
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        all_text[pdf_path] = text
    return all_text

pdf_files = [
    "pdfs/goog-10-k-2023.pdf", 
    "pdfs/tsla-20231231-gen.pdf",    
    "pdfs/uber-10-k-2023.pdf"       
]

pdf_texts = extract_text_from_multiple_pdfs(pdf_files)

