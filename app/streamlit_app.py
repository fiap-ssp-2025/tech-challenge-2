import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.load_model import load_model
from src.explain.shap_explainer import get_shap_values
from src.llm.llm_explainer import explain_case

st.title("Sistema de Avaliação de Risco de Violência Doméstica")

artifact = load_model()

model = artifact["model"]
expected_features = artifact["features"]

# exemplo simples de inputs
idade = st.slider("Idade da vítima", 10, 80, 30)

relacao = st.selectbox(
    "Relação com agressor", ["parceiro_intimo", "familiar", "conhecido", "outros"]
)

viol_psico = st.checkbox("Violência psicológica")
ameaca = st.checkbox("Ameaça")
alcool = st.checkbox("Uso de álcool pelo agressor")


def build_prediction_row(expected_features, idade, relacao, viol_psico, ameaca, alcool):
    """Monta uma linha com todas as features esperadas pelo modelo (ordem e nomes)."""
    row = {f: 0 for f in expected_features}
    # Mapeamento formulário -> feature do modelo (nomes do SINAN/treino)
    if "AG_AMEACA" in row:
        row["AG_AMEACA"] = 1 if ameaca else 0
    if "AUTOR_ALCO" in row:
        row["AUTOR_ALCO"] = 1 if alcool else 0
    if "VIOL_PSICO" in row:
        row["VIOL_PSICO"] = 1 if viol_psico else 0
    if "IDADE" in row:
        row["IDADE"] = idade
    # Situação conjugal / relação: 0=outros, 1=parceiro_intimo, 2=familiar, 3=conhecido (exemplo)
    relacao_code = {"parceiro_intimo": 1, "familiar": 2, "conhecido": 3, "outros": 0}.get(
        relacao, 0
    )
    if "SIT_CONJUG" in row:
        row["SIT_CONJUG"] = relacao_code
    return row


if st.button("Avaliar risco"):

    row = build_prediction_row(
        expected_features, idade, relacao, viol_psico, ameaca, alcool
    )
    data = pd.DataFrame([row], columns=expected_features)

    prob = model.predict_proba(data)[0][1]

    st.subheader(f"Probabilidade de recorrência: {prob:.2f}")

    shap_values = get_shap_values(model, data)

    # Garantir 1D por amostra: (n_samples, n_features) -> primeira linha; (n_features,) -> como está
    sv = np.asarray(shap_values)
    sv_row = sv[0] if sv.ndim > 1 else sv
    sv_row = np.asarray(sv_row).reshape(-1)
    n_features = len(expected_features)
    top_idx = [int(i) for i in np.argsort(np.abs(sv_row))[::-1] if int(i) < n_features][:3]
    top_features = [
        (expected_features[i], float(np.asarray(sv_row[i]).item())) for i in top_idx
    ]

    st.write("Principais fatores:", top_features)

    st.subheader("Explicação com IA")
    try:
        explanation = explain_case(prob, top_features)
        st.write(explanation)
    except Exception as e:
        st.error(
            "Não foi possível gerar a explicação com IA. Verifique a chave OPENAI_API_KEY no .env e sua conexão."
        )
        st.caption(str(e))
