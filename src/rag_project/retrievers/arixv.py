from langchain_arxiv import ArxivRetriever

#create the retriever
retriever = ArxivRetriever(
    k=2, #numbers of papers to retrieve
)

#query arxiv
docs = retriever.invoke("transformer")


# print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("title"))
    print("Authors:", doc.metadata.get("authors"))
    print("Summary:", doc.page_content[:500]) #print first 500 chars