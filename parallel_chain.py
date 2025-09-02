from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_anthropic import ChatAnthropic

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

prompt2=PromptTemplate(
    template='generate a 3 pointer summary from the following text \n {text}',

    input_variables=['text']
)
llm1= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)

llm2= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)
prompt1=PromptTemplate(
    template='Generate a short and simple notes from the following {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate a 5 short quiz questions from the following \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='merge the provided notes and quiz into a single document \n notes-> {notes} \n quiz->{quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | llm1,
        "quiz": prompt2 | llm2
    }
)


merge_chain=prompt3 |llm1 | parser
chain=parallel_chain | merge_chain 
text="""Formula One (F1) is the highest level of international motorsport governed by the FIA, featuring single-seater, open-wheel cars that compete in the annual Formula One World Championship. A typical season includes around 20 to 24 races, known as Grands Prix, held in different countries across the world. The cars use advanced hybrid power units consisting of a 1.6L V6 turbocharged engine combined with an energy recovery system, enabling them to reach speeds of about 350 km/h and accelerate from 0 to 100 km/h in under 2.6 seconds. Aerodynamics play a vital role, with wings, diffusers, and the Drag Reduction System (DRS) improving performance and stability. Drivers face extreme physical challenges, often experiencing G-forces up to 5G during braking, cornering, and acceleration, which demands excellent strength and endurance. Famous teams include Ferrari, Mercedes, Red Bull Racing, McLaren, Aston Martin, and Alpine, with each team fielding two drivers who compete for both the Drivers’ and Constructors’ Championships. Races usually last between 1.5 to 2 hours, covering about 305 km, where strategy, tire management, and pit stops lasting just a few seconds often decide the outcome. F1 is not only a sport of speed but also a blend of advanced engineering, precision teamwork, and global competition

"""

response=chain.invoke({'text':text})
print(response)
chain.get_graph().print_ascii()