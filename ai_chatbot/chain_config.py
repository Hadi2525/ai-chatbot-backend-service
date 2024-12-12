from langchain import hub
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langgraph.graph import START, StateGraph
from langchain_openai import ChatOpenAI
from ai_chatbot.collection_config import get_vectorstore
from typing_extensions import List, TypedDict


prompt = hub.pull("rlm/rag-prompt")

llm = OllamaLLM(
    model="llama3.2:1b",
    temperature=0
)
# import os
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv(),override=True)
# openai_api_key = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=None,

# )

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

async def retrieve(state: State):
    vector_store = await get_vectorstore()
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}

def get_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph


# from IPython.display import Image, display

# image_data = graph.get_graph().draw_mermaid_png()

# with open ("graphFlow.png", "wb") as f:
#     f.write(image_data)

### Testing the LangGraph pipeline ###
# question = "Tell me something about energy saving."
# async def main():
#     result = await graph.ainvoke({"question": question})
#     print(result["answer"])

# import asyncio

# asyncio.run(main())