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



# Create a prompt template
prompt=PromptTemplate(
    template="Write a summary of the following review: {review}",
    input_variables=["review"],
)

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke({"review":"This is a great product. I love it!"})
print(result)

print(chain.get_graph().print_ascii())
