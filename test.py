import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
print(client.models.generate_content(model="gemini-3.5-flash", contents="Say hi").text)
print(SentenceTransformer("all-MiniLM-L6-v2").encode("test").shape)
