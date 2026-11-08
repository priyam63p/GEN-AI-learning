from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001',dimension=32)

documents=[
    "Capital of India is New Delhi",
    "Good Morning"
]

# result=embedding.embed_query("Delhi is the capital of India ")
result=embedding.embed_documents("Delhi is the capital") 
print(str(result))