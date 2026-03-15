import os
from pypdf import PdfReader
from docx import Document
import pandas as pd


def extract_pdf(file_path):

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    return text


def extract_docx(file_path):

    doc = Document(file_path)
    paragraphs = []

    for p in doc.paragraphs:
        if p.text.strip():
            paragraphs.append(p.text)

    return "\n".join(paragraphs)


def extract_excel(file_path):

    xl = pd.ExcelFile(file_path)
    text = ""

    for sheet in xl.sheet_names:

        df = xl.parse(sheet)

        text += f"\nSheet: {sheet}\n"
        text += df.fillna("").to_string()

    return text


def load_documents(folder_path):

    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            content = extract_pdf(path)

        elif file.endswith(".docx"):
            content = extract_docx(path)

        elif file.endswith(".xlsx"):
            content = extract_excel(path)

        else:
            continue

        documents.append({
            "source": file,
            "content": content
        })

    return documents