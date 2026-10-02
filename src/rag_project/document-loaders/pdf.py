from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

file_path = Path(__file__).parent / "NIPS-2017.pdf"
# print(str(file_path))
data = PyPDFLoader(str(file_path))

docs = data.load()

print(len(docs)) #11, it means we have list of 11 documents(creates document as each page of pdf) , inside document with meta-datasource and page_content
print(docs[10])


splitter = TokenTextSplitter(  #or RecursiveCharacterTextSplitter
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)
print(len(chunks)) #and for each chunk there will be 100 token
print(chunks[0].page_content)