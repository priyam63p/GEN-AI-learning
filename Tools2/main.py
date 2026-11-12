import os
import requests
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated

@tool
def get_conversion_factor(base_currency:str,target_currency:str)->float:
    '''
    This function fetches the currency conversion factor between a given base currency and a target currency
    '''
    api_key = os.environ["EXCHANGE_RATE_API_KEY"]
    url = f'https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate


get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})

convert.invoke({'base_currency_value':10, 'conversion_rate':85.16})



# tool binding
llm = ChatGoogleGenerativeAI(model='gemini-3.8-flash')

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]

ai_message = llm_with_tools.invoke(messages)

print(messages.append(ai_message))

print(ai_message.tool_calls)


import json

for tool_call in ai_message.tool_calls:
  # execute the 1st tool and get the value of conversion rate
  if tool_call['name'] == 'get_conversion_factor':
    tool_message1 = get_conversion_factor.invoke(tool_call)
    # fetch this conversion rate
    conversion_rate = json.loads(tool_message1.content)['conversion_rate']
    # append this tool message to messages list
    messages.append(tool_message1)
  # execute the 2nd tool using the conversion rate from tool 1
  if tool_call['name'] == 'convert':
    # fetch the current arg
    tool_call['args']['conversion_rate'] = conversion_rate
    tool_message2 = convert.invoke(tool_call)
    messages.append(tool_message2)

print(messages)

print(llm_with_tools.invoke(messages).content)

