def validate_response(response: str) -> dict:
    forbidden_terms = [
        "não procure ajuda",
        "ignore a ameaça",
        "volte para casa",
        "confronte o agressor",
        "isso não é grave",
        "não conte para ninguém",
        "tome este medicamento",
        "prescrevo",
        "diagnóstico definitivo"
    ]

    response_lower = response.lower()

    found_terms = [
        term for term in forbidden_terms
        if term in response_lower
    ]

    approved = len(found_terms) == 0

    return {
        "approved": approved,
        "found_terms": found_terms,
        "reason": "Resposta segura" if approved else "Resposta contém orientação inadequada"
    }


def add_safety_notice(response: str) -> str:
    notice = (
        "\n\n⚠️ Aviso de segurança: este assistente oferece apoio informativo "
        "e não substitui atendimento de profissionais de saúde, segurança pública, "
        "assistência social ou emergência."
    )

    return response + notice