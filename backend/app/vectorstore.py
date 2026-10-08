import os
import time
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import EMBEDDING_MODEL, PINECONE_INDEX_NAME


@lru_cache
def embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True}
    )


@lru_cache
def pc() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index():
    if not pc().has_index(PINECONE_INDEX_NAME):
        dim = len(embeddings().embed_query("dimension check"))
        pc().create_index(
            name=PINECONE_INDEX_NAME, dimension=dim, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc().describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)


def _index():
    return pc().Index(PINECONE_INDEX_NAME)


def store(user_id: str) -> PineconeVectorStore:
    # One namespace per user = hard isolation between customers.
    return PineconeVectorStore(index=_index(), embedding=embeddings(), namespace=user_id)


def add_docs(user_id, docs, ids):
    store(user_id).add_documents(docs, ids=ids)


def delete_ids(user_id, ids):
    for i in range(0, len(ids), 500):
        _index().delete(ids=ids[i:i + 500], namespace=user_id)


def search(user_id, query, doc_ids, k):
    # Extra metadata filter so a chat only searches the docs the user selected.
    return store(user_id).similarity_search(
        query, k=k, filter={"doc_id": {"$in": doc_ids}}
    )
