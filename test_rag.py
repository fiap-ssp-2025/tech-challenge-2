from src.rag.retriever import build_vectorstore, search_protocol

vectorstore = build_vectorstore(
    "data/protocols/protocolo_violencia.txt"
)

results = search_protocol(
    vectorstore,
    "ameaça e violência doméstica"
)

print(results)