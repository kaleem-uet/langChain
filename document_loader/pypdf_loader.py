from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
load_dotenv()

# Azure OpenAI Model
model = AzureChatOpenAI( 
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Summarize the following article in 5 bullet points {document}',
    input_variables=['document'],
)

loader=PyPDFLoader("article.pdf")

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
print(len(docs))

chain = prompt | model | parser


result=chain.invoke({"document": docs[0].page_content})
print(result)