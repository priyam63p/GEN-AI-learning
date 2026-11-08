from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2',dimension=300)
documents=[
    "Exploring machine learning models on low-spec hardware opens up creative ways to optimize performance.",
    "Managing system resources efficiently prevents memory overflows during local executions",
    "Leveraging lightweight parameters ensures smooth experimentation without hardware bottlenecks",
    "Building intelligent applications requires balancing model complexity with available computing power"
]
query="Tell me about the best college in India"
doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)
print(cosine_similarity([query_embedding],doc_embeddings))
scores=cosine_similarity([query_embedding],doc_embeddings)[0]
# print(sorted(list(enumerate(scores)),key=lambda x:x[1]))

index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(documents[index])
print("Similarity score:",[score])