from retrieval_config import get_vectorstore

async def lookup_contexts(message):
    """
    function is used to lookup contexts from the vector store.
    """
    retriever = await get_vectorstore()

    retrieved_contexts = retriever.similarity_search(message)
    return retrieved_contexts

# import asyncio

# message = "tell me something about the energy saving"

# retrieved = asyncio.run(lookup_contexts(message=message))

# print(retrieved)