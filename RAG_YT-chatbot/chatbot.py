# import os
# os.environ["GEMINI_API_KEY"] = ""


from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings



video_id = "Gfr50f6ZBvo" # only the ID, not full URL

try:
    # 1. Instantiate the API client
    api = YouTubeTranscriptApi()

    # 2. Fetch the transcript using the instance method
    transcript_list = api.fetch(video_id, languages=["en"])

    # 3. Flatten it to plain text (supports the latest snippet object structure)
    transcript = " ".join(snippet.text for snippet in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
except Exception as e:
    print(f"An error occurred: {e}")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])


print(len(chunks))


import time
from langchain_community.vectorstores import FAISS

# Safer settings for the Free Tier rate limits
batch_size = 30
vector_store = None

for i in range(0, len(chunks), batch_size):
  batch = chunks[i : i + batch_size]

  if vector_store is None:
    vector_store = FAISS.from_documents(batch, embeddings)
  else:
    vector_store.add_documents(batch)

  print(
      f"Processed batch {i // batch_size + 1} ({min(i + batch_size, len(chunks))}"
      f"/{len(chunks)} chunks)..."
  )

  # Wait 25 seconds between batches to ensure we don't exceed 100 req/min
  if i + batch_size < len(chunks):
    time.sleep(25)

print("Vector store created successfully!")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

print(retriever.invoke('What is deepmind'))


llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})

print(final_prompt)

answer = llm.invoke(final_prompt)
print(answer.content)


from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})


parallel_chain.invoke('who is Demis')


parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke('Can you summarize the video')