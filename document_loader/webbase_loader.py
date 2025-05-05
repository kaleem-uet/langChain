from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://priceoye.pk/laptops/apple/apple-macbook-air-15-mc7c4-m4-chip")

docs = loader.load()
print(docs[0].page_content)