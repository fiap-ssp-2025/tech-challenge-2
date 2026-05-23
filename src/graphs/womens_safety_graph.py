from typing import TypedDict, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END

from src.guardrails.safety_guardrails import validate_response, add_safety_notice
from src.rag.retriever import build_vectorstore, search_protocol


class SafetyState(TypedDict, total=False):
    user_input: str
    flow_type: str
    risk_level: str
    risk_factors: List[str]
    recommended_actions: List[str]
    protocol_context: str
    explanation: str
    audit_log: Dict[str, Any]
    response: str


def classify_flow(state: SafetyState):
    text = state["user_input"].lower()

    violence_terms = [
        "ameaça", "agressão", "medo", "controle", "violência",
        "apanhei", "me bateu", "me ameaçou", "perseguição",
        "ciúme excessivo", "não deixa sair", "isolamento",
        "arma", "faca", "revólver", "morte"
    ]

    psychosocial_terms = [
        "ansiedade", "ansiosa", "pânico", "depressão",
        "chorando", "tristeza", "desespero",
        "não consigo dormir", "crise"
    ]

    reproductive_terms = [
        "gravidez", "grávida", "gestante", "pós-parto", "aborto",
        "sangramento", "pré-natal", "amamentação"
    ]

    if any(term in text for term in violence_terms):
        flow = "deteccao_violencia_contra_mulher"
    elif any(term in text for term in psychosocial_terms):
        flow = "acolhimento_psicossocial"
    elif any(term in text for term in reproductive_terms):
        flow = "saude_mulher_contextual"
    else:
        flow = "orientacao_preventiva"

    return {"flow_type": flow}


def assess_risk(state: SafetyState):
    text = state["user_input"].lower()

    risk_rules = {
        "ameaça de morte": "Relato de ameaça de morte",
        "arma": "Possível presença de arma",
        "faca": "Possível presença de arma branca",
        "revólver": "Possível presença de arma de fogo",
        "tenho medo": "Medo declarado da vítima",
        "não consigo sair": "Indício de cárcere, controle ou restrição de liberdade",
        "me controla": "Controle coercitivo",
        "controle": "Possível controle coercitivo",
        "me bateu": "Violência física relatada",
        "agressão física": "Violência física relatada",
        "perseguição": "Possível perseguição/stalking",
        "sangramento": "Sintoma físico alarmante",
        "grávida": "Condição de maior vulnerabilidade",
        "gravidez": "Condição de maior vulnerabilidade",
        "gestante": "Condição de maior vulnerabilidade",
        "ansiosa": "Sinal de sofrimento emocional",
        "ansiedade": "Sinal de sofrimento emocional",
        "chorando": "Sinal de sofrimento emocional",
        "não consigo dormir": "Sinal de sofrimento emocional",
    }

    factors = list(set([
        description
        for term, description in risk_rules.items()
        if term in text
    ]))

    very_high_terms = [
        "ameaça de morte", "arma", "revólver", "faca",
        "não consigo sair", "sangramento"
    ]

    high_terms = [
        "tenho medo", "me bateu", "agressão física",
        "perseguição", "me controla", "controle",
        "grávida", "gravidez", "gestante"
    ]

    moderate_terms = [
        "ansiosa", "ansiedade", "chorando",
        "não consigo dormir", "tristeza", "desespero"
    ]

    if any(term in text for term in very_high_terms):
        risk = "crítico"
    elif any(term in text for term in high_terms):
        risk = "alto"
    elif any(term in text for term in moderate_terms):
        risk = "moderado"
    elif factors:
        risk = "moderado"
    else:
        risk = "baixo"

    return {
        "risk_level": risk,
        "risk_factors": factors
    }


def retrieve_protocol_context(state: SafetyState):
    flow = state["flow_type"]

    protocol_map = {
        "deteccao_violencia_contra_mulher": [
            "data/protocols/protocolo_violencia.txt",
            "data/protocols/lei_maria_da_penha.txt",
            "data/protocols/canais_apoio_mulher_df.txt",
            "data/protocols/protocolo_rede_protecao_df.txt",
            "data/protocols/lgpd_dados_sensiveis.txt",
        ],
        "acolhimento_psicossocial": [
            "data/protocols/protocolo_saude_mental.txt",
            "data/protocols/canais_apoio_mulher_df.txt",
            "data/protocols/lgpd_dados_sensiveis.txt",
        ],
        "saude_mulher_contextual": [
            "data/protocols/protocolo_obstetrico.txt",
            "data/protocols/lgpd_dados_sensiveis.txt",
        ],
        "orientacao_preventiva": [
            "data/protocols/protocolo_prevencao.txt",
            "data/protocols/canais_apoio_mulher_df.txt",
            "data/protocols/lei_maria_da_penha.txt",
        ],
    }

    selected_files = protocol_map.get(
        flow,
        ["data/protocols/protocolo_prevencao.txt"]
    )

    contexts = []

    for file_path in selected_files:
        vectorstore = build_vectorstore(file_path)
        results = search_protocol(vectorstore, state["user_input"])

        for result in results:
            clean_result = result.strip()
            if clean_result and clean_result not in contexts:
                contexts.append(clean_result)

    context = "\n\n".join(contexts)

    return {"protocol_context": context}


def define_recommended_actions(state: SafetyState):
    risk = state["risk_level"]
    flow = state["flow_type"]

    actions = []

    if flow == "deteccao_violencia_contra_mulher":
        actions.extend([
            "Realizar acolhimento humanizado e escuta qualificada.",
            "Preservar sigilo, privacidade e confidencialidade das informações.",
            "Avaliar se há risco imediato à integridade física da mulher.",
            "Orientar busca de apoio profissional especializado na rede de proteção.",
            "Evitar qualquer conduta que exponha a mulher a novo risco."
        ])

        if risk in ["crítico", "alto"]:
            actions.extend([
                "Sugerir acionamento imediato de serviço de emergência, se houver perigo atual.",
                "Encaminhar para equipe especializada de segurança, saúde ou assistência social.",
                "Registrar o caso de forma segura, com restrição de acesso a dados sensíveis."
            ])

    elif flow == "acolhimento_psicossocial":
        actions.extend([
            "Realizar escuta acolhedora e sem julgamento.",
            "Sugerir atendimento com equipe psicossocial.",
            "Avaliar sinais de sofrimento intenso ou risco à própria integridade.",
            "Encaminhar para serviço especializado quando necessário.",
            "Preservar confidencialidade do relato."
        ])

    elif flow == "saude_mulher_contextual":
        actions.extend([
            "Orientar avaliação presencial com profissional de saúde.",
            "Não realizar diagnóstico definitivo.",
            "Não prescrever medicamentos.",
            "Em caso de sintomas alarmantes, sugerir atendimento imediato."
        ])

    else:
        actions.extend([
            "Fornecer orientação preventiva geral.",
            "Sugerir busca de serviço especializado para avaliação individual.",
            "Reforçar que o assistente não substitui profissional habilitado."
        ])

    return {"recommended_actions": actions}


def generate_explanation(state: SafetyState):
    factors = state.get("risk_factors", [])

    if factors:
        factor_text = "; ".join(factors)
    else:
        factor_text = "Não foram identificados sinais explícitos de risco elevado no relato informado."

    explanation = (
        f"O fluxo foi classificado como '{state['flow_type']}' com nível de risco "
        f"'{state['risk_level']}'. A classificação considerou os seguintes fatores: "
        f"{factor_text}. A resposta foi construída com apoio do protocolo recuperado "
        f"pelo módulo RAG e submetida ao módulo de validação de segurança."
    )

    return {"explanation": explanation}


def create_audit_log(state: SafetyState):
    log = {
        "timestamp": datetime.now().isoformat(),
        "flow_type": state.get("flow_type"),
        "risk_level": state.get("risk_level"),
        "risk_factors": state.get("risk_factors", []),
        "used_rag": bool(state.get("protocol_context")),
        "sensitive_case": state.get("flow_type") == "deteccao_violencia_contra_mulher",
        "safety_validation": "pending"
    }

    return {"audit_log": log}


def generate_response(state: SafetyState):
    actions = state.get("recommended_actions", [])
    actions_text = "\n".join([f"- {action}" for action in actions])

    factors = state.get("risk_factors", [])
    factors_text = (
        "\n".join([f"- {factor}" for factor in factors])
        if factors
        else "- Nenhum fator crítico explícito identificado."
    )

    response = f"""
Fluxo identificado:
{state['flow_type']}

Nível de risco:
{state['risk_level']}

Fatores considerados:
{factors_text}

Contexto recuperado do protocolo:
{state['protocol_context']}

Orientações recomendadas:
{actions_text}

Explicação da decisão:
{state['explanation']}

Fonte:
Protocolos internos, documentos normativos, módulo RAG e regras de segurança do sistema.

Limitações:
Este assistente atua como apoio à triagem e à decisão. Ele não substitui atendimento profissional, não realiza diagnóstico definitivo e não deve ser usado como único instrumento de decisão em casos sensíveis.
"""

    validation = validate_response(response)

    audit_log = state.get("audit_log", {})
    audit_log["safety_validation"] = "approved" if validation["approved"] else "blocked"
    audit_log["blocked_terms"] = validation.get("found_terms", [])

    if not validation["approved"]:
        response = (
            "A resposta foi bloqueada pelo módulo de segurança por conter "
            "orientação potencialmente inadequada para um caso sensível."
        )

    response = add_safety_notice(response)

    return {
        "response": response,
        "audit_log": audit_log
    }


def build_graph():
    graph = StateGraph(SafetyState)

    graph.add_node("classify_flow", classify_flow)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("retrieve_protocol_context", retrieve_protocol_context)
    graph.add_node("define_recommended_actions", define_recommended_actions)
    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("create_audit_log", create_audit_log)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("classify_flow")

    graph.add_edge("classify_flow", "assess_risk")
    graph.add_edge("assess_risk", "retrieve_protocol_context")
    graph.add_edge("retrieve_protocol_context", "define_recommended_actions")
    graph.add_edge("define_recommended_actions", "generate_explanation")
    graph.add_edge("generate_explanation", "create_audit_log")
    graph.add_edge("create_audit_log", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()