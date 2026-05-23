from src.graphs.womens_safety_graph import build_graph

graph = build_graph()

result = graph.invoke({
    "user_input": "Tenho medo do meu parceiro. Ele me ameaça e controla minhas consultas."
})

print(result)