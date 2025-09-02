from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
prompt=PromptTemplate(
    template='Generate five interesting facts about {topic}',
    input_variables=['topic']
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)


parser=StrOutputParser()
chain=prompt | llm | parser 
response=chain.invoke({'topic':'F1 racing'})
print(response)

chain.get_graph().print_ascii()
