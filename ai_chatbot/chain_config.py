import os

from dotenv import find_dotenv, load_dotenv
from langchain import hub
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from ai_chatbot.collection_config import get_query_results

OAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONN_STRING = os.getenv("CONN_STRING2")
DATABASE_NAME = "ai_chatbot"
COLLECTION_NAME = "semantic_cache"
INDEX_NAME = "vector_embeddings"
_ = load_dotenv(find_dotenv(), override=True)
model = "text-embedding-3-small"

prompt = hub.pull("rlm/rag-prompt")

llm = OllamaLLM(model="llama3.2:1b", temperature=0)
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


def retrieve(state: State):
    retrieved_docs = get_query_results(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


def get_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    setup_semantic_cache()
    graph = graph_builder.compile()
    return graph


def setup_semantic_cache():
    try:
        embeddings = OpenAIEmbeddings(
        model=model, api_key=OAI_API_KEY
        )

        set_llm_cache(
            MongoDBAtlasSemanticCache(
                connection_string=CONN_STRING,
                database_name=DATABASE_NAME,
                collection_name=COLLECTION_NAME,
                embedding=embeddings,
                index_name=INDEX_NAME,
                score_threshold=0.95
            )
        )
        return True
    except Exception as e:
        print(e)
        return False

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
