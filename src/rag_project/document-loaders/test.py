#import document_loaders to load any type of document
from pathlib import Path
from langchain_community.document_loaders import TextLoader


file_path = Path(__file__).parent / "notes.txt"
data = TextLoader(str(file_path)) #TextLoader object
docs = data.load()

print(docs) #We get list , inside that we have our document with metadata (source) and page_content
print(len(docs)) #1
# print(docs[0]) #get the document 1st content and the metadata
print(docs[0].page_content)