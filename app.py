import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def load():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
    return model, col

model, col = load()

st.title("?? Document Q&A Chatbot")
st.caption("Ask questions about your PDFs. Answers come with sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    res = col.query(query_embeddings=[model.encode(q).tolist()], n_results=4)
    docs, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(f"[{m['source']} p.{m['page']}]\n{d}" for d, m in zip(docs, metas))
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}"
    )
    with st.chat_message("assistant"):
        try:
            answer = client.models.generate_content(model="gemini-3.5-flash", contents=prompt).text
        except Exception as e:
            answer = f"Model error, try again in a moment. ({type(e).__name__})"
        sources = sorted({f"{m['source']} p.{m['page']}" for m in metas})
        full = answer + "\n\n**Sources:** " + ", ".join(sources)
        st.markdown(full)
    st.session_state.messages.append({"role": "assistant", "content": full})
