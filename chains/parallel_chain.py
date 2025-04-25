from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()

# Azure OpenAI Model
azure_model = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Groq Model
groq_model = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192",
    temperature=0,
)

# Prompts
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n{text}",
    input_variables=["text"]
)
prompt3 = PromptTemplate(
    template="Generate a quiz from the following details and summary:\nDetails:\n{details}\nSummary:\n{summary}",
    input_variables=["details", "summary"]
)
parser = StrOutputParser()

# Build parallel chain
parallel_chain = RunnableParallel({
    "details": prompt1 | azure_model | parser,
    "summary": prompt2 | groq_model | parser,
})

# Merge chain (compose quiz)
merge_chain = prompt3 | azure_model | parser

# Full pipeline: Parallel step --> Merge step
chain = parallel_chain | merge_chain

# Example input
topic = "The benefits of regular exercise"

# Run parallel step alone (optional, for demonstration)
intermediate = parallel_chain.invoke({"topic": topic, "text": topic})
print("=== PARALLEL OUTPUT ===")
print(intermediate)
print("")

# Run full pipeline (parallel -> merge/quiz)
result = chain.invoke({"topic": topic, "text": topic})
print("=== FINAL MERGED RESULT ===")
print(result)

print("\n=== PIPELINE GRAPH ===")
chain.get_graph().print_ascii()
