from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch,RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm1= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)

llm2= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)
parser=StrOutputParser()
class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Give the sentiment of the feedback')
parser2=PydanticOutputParser(pydantic_object=Feedback)   

prompt1=PromptTemplate(
    template='Classify the sentiment of following feedback text into positive and negative  \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template="Write a polite and concise response to this negative feedback: {feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template='Write appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']

)

classifier_chain=prompt1 | llm1 |parser2

branch_chain=RunnableBranch(
    (lambda x:x.sentiment =='positive',prompt3|llm2|parser),
    (lambda x:x.sentiment =='negative',prompt2|llm2|parser),
    RunnableLambda(lambda x:"could not find sentiment")
    
)

chain=classifier_chain | branch_chain
response=chain.invoke({'feedback':'This is a bad race of ferrari today'})
print(response)
chain.get_graph().print_ascii()