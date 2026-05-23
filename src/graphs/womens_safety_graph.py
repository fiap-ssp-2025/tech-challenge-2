from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.guardrails.safety_guardrails import validate_response, add_safety_notice

class SafetyState(TypedDict):
    user_input: str
    flow_type: str
    risk_level: str
    response: str


def classify_flow(state: SafetyState):

    text = state["user_input"].lower()

    if any(term in text for term in [
        "ameaça",
        "agressão",
        "medo",
        "controle",
        "violência"
    ]):
        flow = "violencia_domestica"

    elif any(term in text for term in [
        "ansiedade",
        "pânico",
        "depressão"
    ]):
        flow = "acolhimento_psicossocial"

    elif any(term in text for term in [
        "gravidez",
        "gestante",
        "pós-parto"
    ]):
        flow = "saude_mulher_contextual"

    else:
        flow = "orientacao_preventiva"

    return {"flow_type": flow}


def assess_risk(state: SafetyState):

    text = state["user_input"].lower()

    high_risk_terms = [
        "ameaça de morte",
        "arma",
        "agressão física",
        "tenho medo",
        "não consigo sair",
        "sangramento"
    ]

    risk = "alto" if any(term in text for term in high_risk_terms) else "moderado"

    return {"risk_level": risk}


def generate_response(state: SafetyState):

    response = f"""
Fluxo identificado: {state['flow_type']}

Nível de risco: {state['risk_level']}

Orientação inicial:
- Realizar acolhimento seguro
- Preservar confidencialidade
- Avaliar necessidade de encaminhamento
- Em caso de risco imediato, procurar atendimento emergencial

Fonte: regras internas de segurança e acolhimento.
"""

    validation = validate_response(response)

    if not validation["approved"]:
        response = (
            "A resposta foi bloqueada pelo módulo de segurança por conter "
            "orientação inadequada."
        )

    response = add_safety_notice(response)

    return {"response": response}

def build_graph():

    graph = StateGraph(SafetyState)

    graph.add_node("classify_flow", classify_flow)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("classify_flow")

    graph.add_edge("classify_flow", "assess_risk")
    graph.add_edge("assess_risk", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()