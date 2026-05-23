import hashlib
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


_EMBEDDINGS = None


def get_embeddings():
    global _EMBEDDINGS

    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    return _EMBEDDINGS


def _collection_name_from_path(file_path: str) -> str:
    file_name = Path(file_path).stem
    file_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
    return f"{file_name}_{file_hash}"


def build_vectorstore(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de protocolo não encontrado: {file_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Arquivo de protocolo vazio: {file_path}")

    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(f"Nenhum chunk foi gerado para o arquivo: {file_path}")

    embeddings = get_embeddings()

    collection_name = _collection_name_from_path(file_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="data/chroma_db",
        collection_name=collection_name
    )

    return vectorstore


def search_protocol(vectorstore, query: str, k: int = 1):
    results = vectorstore.similarity_search(query, k=k)

    unique_results = []
    for doc in results:
        content = doc.page_content.strip()
        if content and content not in unique_results:
            unique_results.append(content)

    return unique_results