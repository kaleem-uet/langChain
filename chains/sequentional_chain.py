from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


prompt1=PromptTemplate(
    
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({"topic":"Un employment in pakistan"})
print(result)

chain.get_graph().print_ascii()