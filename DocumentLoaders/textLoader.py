from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv
model=ChatGoogleGenerativeAI(model='gemini-3.8-flash')

prompt=PromptTemplate(
    template='Write a summary for the follwong poem {poem}',
    input_variables=['poem']
)


parser=StrOutputParser()


loader=TextLoader('sample.txt',encoding='utf-8')
docs=loader.load()
# print(docs)
# print(type(docs))
# print(len(docs))
# print(docs[0])

chain=prompt|model|parser
print(chain.invoke({'poem':docs[0].page_content}))
