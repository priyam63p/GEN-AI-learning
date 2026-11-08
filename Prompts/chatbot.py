from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv

load_dotenv()
model = GoogleGenerativeAI(model="gemini-3.8-flash")
chat_history=[
    SystemMessage(content="You are a professional AI Engineer")
]
while True:
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result=model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.text)