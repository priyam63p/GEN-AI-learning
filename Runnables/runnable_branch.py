from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda,RunnableBranch


load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.8-flash')



prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Summarize the follwoing text - {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1,model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))