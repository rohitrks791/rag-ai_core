from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/iphone-duo/"

data = WebBaseLoader(url)
docs = data.load()

print(len(docs)) # 1 since for every page we get 1
