from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

# Azure OpenAI Model
azure_model = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


