import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

q = input("Ask something from your PDF: ")
res = col.query(query_embeddings=[model.encode(q).tolist()], n_results=3)
for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
    print(f"\n[{meta['source']} | page {meta['page']}]\n{doc[:300]}")
