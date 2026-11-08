from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

chat_template=ChatPromptTemplate([
    # SystemMessage(content='You are a helpful {domain}  expert'),
    # HumanMessage(content='Explain in simple terms, what is {topic}')
    ('system','You are a helpful {domain} expert'),
    ('human','Explain in simple terms what is {domain}'),
])

prompt=chat_template.invoke({'domain':'cricket','topic':'Dusra'})
print(prompt)