from langchain_community.document_loaders import TextLoader

loader = TextLoader("earth.txt", encoding="utf-8")
documents = loader.load()
print(type(documents))