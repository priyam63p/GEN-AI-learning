from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

parser=JsonOutputParser()

template=PromptTemplate(
    template='Give me the name, age,city of a fictional person \n{fromat_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
# prompt=template.format()
# # print(prompt)
# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result)
# print(type(final_result))


chain=template|model|parser
result=chain.invoke({})
print(result)