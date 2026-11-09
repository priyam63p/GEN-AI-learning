from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda


load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.8-flash')
parser=StrOutputParser()
prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

def word_counter(text):
    return len(text.split())
runnable_word_counter=RunnableLambda(word_counter)
runnable_sequence=RunnableSequence(prompt,model,parser)
parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_counter)
})
# print(runnable_word_counter.invoke('Hello how are you?'))

final_chain=RunnableSequence(runnable_sequence,parallel_chain)
print(final_chain.invoke({'topic':'AI'}))