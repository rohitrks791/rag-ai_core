from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

from pathlib import Path
from langchain_community.document_loaders import TextLoader,PyPDFLoader

file_path = Path(__file__).parent / "sample.txt"
# data = PyPDFLoader(str(file_path)) 
data = TextLoader(str(file_path))
docs = data.load()
text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

chunks = text_splitter.split_documents(docs)
print("Original documents:", len(docs))
print("Total chunks:", len(chunks))
# print(chunks)

for i in chunks:
    print(i)
    print()

# for i, chunk in enumerate(chunks[:5]):
#     print(f"\n--- Chunk {i + 1} ---")
#     print(chunk.page_content)
