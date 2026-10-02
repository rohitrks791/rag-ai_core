from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate # bcz with this we can give roles as well
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = Path(__file__).parent / "NIPS-2017.pdf"
#But here we can get problem of context window[at once we can only send limited amount data to llm]
data = PyPDFLoader(str(file_path))
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000 ,
    chunk_overlap = 200,
)
chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a AI that summarizes the text"),
        ("human", "{data}")
    ]
)

llm = ChatNVIDIA(
        model="nvidia/nemotron-3-super-120b-a12b",
        temperature=0.6
       )
# prompt = template.format_messages(data = docs[0].page_content) or
# prompt = template.invoke({"data": docs[0].page_content}) #here we are only document first one
prompt = template.invoke({"data": docs}) #So we are also able to send whole pdf data to llm for processing , but for pdf that contains around 150-200 more pages then we will not send directly
result = llm.invoke(prompt) # We are feeding external data sources to the llm
print(result.content)