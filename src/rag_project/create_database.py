#load pdf
#split into chunks
#create the embeddings
#store into chroma
#data injection: We want to take the all data at once.
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

file_path = Path(__file__).parent /"document-loaders"/"NIPS-2017.pdf"
print(file_path)
# exit
#But here we can get problem of context window[at once we can only send limited amount data to llm]
data = PyPDFLoader(str(file_path))
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
    )

#Now vector store will be created witht this
vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory= "chroma_db"
)