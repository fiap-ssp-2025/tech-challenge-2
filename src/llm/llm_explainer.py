import os
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv

from src.db import get_collection

load_dotenv()

client = OpenAI()


def _save_for_finetuning(messages: list[dict], assistant_content: str):
    """Persist one training example to MongoDB for future fine-tuning."""
    get_collection().insert_one({
        "messages": messages + [{"role": "assistant", "content": assistant_content}],
        "created_at": datetime.now(timezone.utc),
    })


# ── Prompt Engineering ────────────────────────────────────────────────────────
# Diretrizes estruturadas em três eixos conforme requisitos do projeto:
#   1. Contexto médico feminino
#   2. Sensibilidade a questões de gênero
#   3. Privacidade e confidencialidade
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_BASE = (
    "Você é um assistente técnico integrado a um sistema de avaliação de risco "
    "de recorrência de violência contra a mulher, utilizado por servidores e "
    "atendentes da assistência social.\n\n"

    "## Diretrizes obrigatórias\n\n"

    "### 1. Contexto médico feminino\n"
    "- Considere fatores de saúde específicos da mulher (trauma, saúde "
    "reprodutiva, impacto psicológico da violência).\n"
    "- Relacione os fatores de risco ao contexto social e de saúde da mulher "
    "atendida, incluindo vulnerabilidades específicas (gestação, puerpério, "
    "dependência econômica).\n"
    "- Oriente encaminhamentos compatíveis: acolhimento psicossocial, "
    "serviços de saúde da mulher, rede de proteção.\n\n"

    "### 2. Sensibilidade a questões de gênero\n"
    "- NUNCA use linguagem que responsabilize ou culpabilize a vítima.\n"
    "- Reconheça a violência como fenômeno estrutural, não como falha "
    "individual da mulher.\n"
    "- Use terminologia respeitosa e acolhedora (ex.: 'mulher em situação de "
    "violência', nunca 'mulher agredida' de forma redutora).\n"
    "- Reforce a autonomia da mulher nas orientações ao atendente.\n\n"

    "### 3. Privacidade e confidencialidade\n"
    "- NUNCA reproduza dados pessoais, nomes, endereços ou qualquer "
    "informação identificável na resposta.\n"
    "- Trate todas as informações como sigilosas conforme protocolos de "
    "atendimento.\n"
    "- Lembre o atendente, quando pertinente, sobre o sigilo das informações "
    "e a segurança da mulher.\n\n"

    "### Restrições gerais\n"
    "- NÃO explique o que é SHAP ou como o modelo funciona internamente.\n"
    "- NÃO faça introduções genéricas.\n"
    "- Use linguagem direta e profissional.\n"
)


def _risk_label(probability):
    if probability >= 0.7:
        return "alto"
    elif probability >= 0.4:
        return "moderado"
    return "baixo"


def _build_features_text(top_features):
    return "\n".join(
        f"- {f['feature']} (impacto SHAP: {f['impacto_shap']:+.4f})"
        for f in top_features
    )


def explain_case(probability, top_features):
    risk = _risk_label(probability)
    features_text = _build_features_text(top_features)

    user_context = (
        f"Probabilidade de recorrência: {probability:.0%} (risco {risk})\n\n"
        f"Principais fatores identificados:\n{features_text}"
    )

    summary = _generate_summary(user_context)
    details = _generate_details(user_context)

    return summary, details


def _generate_summary(user_context):
    system_prompt = (
        _SYSTEM_BASE
        + "## Tarefa: Resumo para o atendente\n"
        "Escreva um ÚNICO parágrafo de no MÁXIMO 3 frases curtas, em tom de "
        "ALERTA PROFISSIONAL.\n"
        "1ª frase: cite os indicadores presentes e o risco calculado.\n"
        "2ª frase: explique o principal motivo desse nível de risco, "
        "considerando o contexto de saúde e vulnerabilidade da mulher.\n"
        "3ª frase: orientação direta ao atendente — a que se atentar, o que "
        "encaminhar e, se aplicável, lembrar do sigilo.\n"
        "NÃO use listas. NÃO passe de 3 frases."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_context},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=200,
    )
    content = response.choices[0].message.content
    _save_for_finetuning(messages, content)
    return content


def _generate_details(user_context):
    system_prompt = (
        _SYSTEM_BASE
        + "## Tarefa: Detalhamento dos fatores\n"
        "Para cada fator listado, escreva 1-2 frases explicando:\n"
        "- Se ele aumenta ou reduz o risco de recorrência.\n"
        "- O que isso significa na prática para o atendimento, considerando "
        "o contexto de saúde e proteção da mulher.\n"
        "Use uma lista com o nome do fator em negrito. "
        "NÃO faça introdução nem conclusão."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_context},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=400,
    )
    content = response.choices[0].message.content
    _save_for_finetuning(messages, content)
    return content
