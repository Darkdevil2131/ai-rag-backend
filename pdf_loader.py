from pypdf import PdfReader

# 🔹 Load PDF and extract text
def load_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# 🔹 Clean text (important for better embeddings)
def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())  # remove extra spaces
    return text


# 🔹 Chunk text into smaller parts
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap for better context

    return chunks


# 🔹 Full pipeline (load → clean → chunk)
def process_pdf(path):
    raw_text = load_pdf(path)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)
    return chunks