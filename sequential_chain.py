from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
prompt1=PromptTemplate(
    template='Generate a detailed replort on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='generate a 3 pointer summary from the following text \n {text}',

    input_variables=['text']
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)

parser=StrOutputParser()
chain=prompt1 | llm | prompt2 | llm | parser
response=chain.invoke({'topic':'Louis Hamilton'})
print(response)
chain.get_graph().print_ascii()