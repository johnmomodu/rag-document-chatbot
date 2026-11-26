\# 📄 Document Q\&A Chatbot (RAG)



Ask questions about your PDFs and get answers with page-level sources. If the answer isn't in the documents, it says so instead of guessing.


\## How it works

1\. PDFs are split into overlapping chunks

2\. Each chunk is embedded with `all-MiniLM-L6-v2` and stored in ChromaDB

3\. A question is embedded and the 4 closest chunks are retrieved

4\. Gemini answers using only those chunks and cites the pages



\## Tech stack

Python, Streamlit, ChromaDB, sentence-transformers, Google Gemini API



\## Run locally

```bash

git clone https://github.com/johnmomodu/rag-document-chatbot

cd rag-document-chatbot

pip install -r requirements.txt

```

Create a `.env` file containing `GOOGLE\_API\_KEY=your\_key`, put PDFs in `docs/`, then:

```bash

python ingest.py

streamlit run app.py

```



\## Author

John Momodu · \[LinkedIn](https://www.linkedin.com/in/john-momodu-330b5b339)

