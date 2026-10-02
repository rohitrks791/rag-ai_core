from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate # bcz with this we can give roles as well

file_path = Path(__file__).parent / "notes.txt"
data = TextLoader(str(file_path)) #TextLoader object
docs = data.load()

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
prompt = template.invoke({"data": docs[0].page_content})
result = llm.invoke(prompt) # We are feeding external data sources to the llm
print(result.content)