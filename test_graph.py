from src.graphs.womens_safety_graph import build_graph


def run_case(case_text: str):
    app = build_graph()

    result = app.invoke({
        "user_input": case_text
    })

    print("\n" + "=" * 80)
    print("CASO TESTADO")
    print("=" * 80)
    print(case_text)

    print("\n" + "=" * 80)
    print("RESULTADO DO ASSISTENTE")
    print("=" * 80)
    print(result["response"])

    print("\n" + "=" * 80)
    print("LOG DE AUDITORIA")
    print("=" * 80)
    print(result["audit_log"])


if __name__ == "__main__":
    cases = [
        "Tenho medo do meu parceiro. Ele me ameaça e controla minhas consultas.",
        "Estou grávida e tive sangramento hoje pela manhã.",
        "Estou muito ansiosa, chorando muito e sem conseguir dormir.",
        "Quero saber como prevenir situações de violência e onde buscar orientação."
    ]

    for case in cases:
        run_case(case)