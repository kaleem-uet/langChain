from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader("../", glob="**/*.md",loader_cls=PyPDFLoader)
docs = loader.lazy_load()
docs_count = len(docs)
print(docs_count)