import glob
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_or_create_collection("docs")

def chunk(text, size=800, overlap=100):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]

for path in glob.glob("docs/*.pdf"):
    for page_no, page in enumerate(PdfReader(path).pages, 1):
        for j, c in enumerate(chunk(page.extract_text() or "")):
            if not c.strip():
                continue
            col.upsert(
                ids=[f"{path}-{page_no}-{j}"],
                documents=[c],
                embeddings=[model.encode(c).tolist()],
                metadatas=[{"source": path, "page": page_no}],
            )

print("Chunks stored:", col.count())
