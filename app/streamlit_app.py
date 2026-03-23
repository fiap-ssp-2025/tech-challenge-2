import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain.shap_explainer import get_shap_values
from src.optimize_ga import load_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from src.llm.llm_explainer import explain_case

st.set_page_config(
    page_title="Avaliação de Risco de Violência Doméstica",
    layout="wide",
)

st.title("Sistema de Avaliação de Risco de Violência Doméstica")

BASE_PATH = Path(__file__).resolve().parent.parent

# escolha do modelo
model_option = st.selectbox(
    "Selecione o modelo",
    [
        ("Balanced", "random_forest_balanced.pkl"),
        ("Current", "random_forest_current.pkl"),
        ("Clinical", "random_forest_clinical.pkl"),
        ("Original", "random_forest.pkl"),
    ],
    format_func=lambda x: x[0],
)

model_label, model_file = model_option

artifact_path = BASE_PATH / "models" / model_file

if not artifact_path.exists():
    st.error(f"Arquivo do modelo não encontrado: {artifact_path}")
    st.stop()

artifact = joblib.load(artifact_path)

model = artifact["model"]
expected_features = artifact["features"]

st.caption(f"Modelo carregado: {model_label}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Parâmetros do modelo")
    st.json(artifact.get("params", {}))

with col2:
    st.subheader("Informações gerais")
    info = {
        "fitness_mode": artifact.get("fitness_mode", "não informado"),
        "fitness": artifact.get("fitness", "não informado"),
        "arquivo": model_file,
    }
    st.json(info)

st.divider()

col_metrics_1, col_metrics_2 = st.columns(2)

with col_metrics_1:
    st.subheader("Métricas do modelo salvo")
    st.json(artifact.get("metrics", {}))

with col_metrics_2:
    st.subheader("Métricas durante a busca")
    st.json(artifact.get("search_metrics", {}))

st.divider()

st.subheader("Simulação de caso individual")

form_col1, form_col2 = st.columns(2)

with form_col1:
    idade = st.slider("Idade da vítima", 10, 80, 30)
    relacao = st.selectbox(
        "Relação com agressor",
        ["parceiro_intimo", "familiar", "conhecido", "outros"],
    )

with form_col2:
    viol_psico = st.checkbox("Violência psicológica")
    ameaca = st.checkbox("Ameaça")
    alcool = st.checkbox("Uso de álcool pelo agressor")


def build_prediction_row(expected_features, idade, relacao, viol_psico, ameaca, alcool):
    row = {f: 0 for f in expected_features}

    if "AG_AMEACA" in row:
        row["AG_AMEACA"] = 1 if ameaca else 0
    if "AUTOR_ALCO" in row:
        row["AUTOR_ALCO"] = 1 if alcool else 0
    if "VIOL_PSICO" in row:
        row["VIOL_PSICO"] = 1 if viol_psico else 0
    if "IDADE" in row:
        row["IDADE"] = idade

    relacao_code = {
        "parceiro_intimo": 1,
        "familiar": 2,
        "conhecido": 3,
        "outros": 0,
    }.get(relacao, 0)

    if "SIT_CONJUG" in row:
        row["SIT_CONJUG"] = relacao_code

    return row


if st.button("Avaliar risco individual"):
    row = build_prediction_row(
        expected_features, idade, relacao, viol_psico, ameaca, alcool
    )
    data = pd.DataFrame([row], columns=expected_features)

    prob = model.predict_proba(data)[0][1]

    st.subheader(f"Probabilidade de recorrência: {prob:.2f}")

    shap_values = get_shap_values(model, data)

    sv = np.asarray(shap_values)
    sv_row = sv[0] if sv.ndim > 1 else sv
    sv_row = np.asarray(sv_row).reshape(-1)
    n_features = len(expected_features)

    top_idx = [
        int(i) for i in np.argsort(np.abs(sv_row))[::-1] if int(i) < n_features
    ][:5]

    top_features = [
        {
            "feature": expected_features[i],
            "impacto_shap": float(np.asarray(sv_row[i]).item()),
        }
        for i in top_idx
    ]

    st.subheader("Principais fatores")
    st.dataframe(pd.DataFrame(top_features), use_container_width=True)

    st.subheader("Explicação com IA")
    st.info("Explicação com IA desativada temporariamente para testes locais.")
    # explanation = explain_case(prob, top_features)
    # st.write(explanation)

st.divider()

st.subheader("Avaliação do modelo")

if st.button(f"Mostrar matriz de confusão do modelo {model_label}"):
    _, _, _, X_test, _, y_test = load_data()

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    st.markdown(f"### Matriz de Confusão - {model_label}")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    st.pyplot(fig)

    summary_df = pd.DataFrame(
        [
            ["Verdadeiros Negativos (TN)", int(tn)],
            ["Falsos Positivos (FP)", int(fp)],
            ["Falsos Negativos (FN)", int(fn)],
            ["Verdadeiros Positivos (TP)", int(tp)],
        ],
        columns=["Tipo", "Valor"],
    )

    st.dataframe(summary_df, use_container_width=True)

st.divider()

st.subheader("Equidade por grupo")

group_metrics = artifact.get("group_metrics", {})

if not group_metrics:
    st.warning("Nenhuma métrica de grupo foi encontrada no artifact salvo.")
else:
    summary = group_metrics.get("_summary", {})
    groups_only = {
        k: v for k, v in group_metrics.items()
        if k != "_summary"
    }

    if summary:
        st.markdown("### Resumo de equidade")
        st.json(summary)

    if groups_only:
        st.markdown("### Métricas por raça")

        rows = []
        for group_name, values in groups_only.items():
            row = {"grupo": group_name}
            row.update(values)
            rows.append(row)

        df_groups = pd.DataFrame(rows)

        preferred_columns = [
            "grupo",
            "size",
            "recall",
            "specificity",
            "precision",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        existing_columns = [c for c in preferred_columns if c in df_groups.columns]
        df_groups = df_groups[existing_columns]

        st.dataframe(df_groups, use_container_width=True)

        if "recall" in df_groups.columns and "grupo" in df_groups.columns:
            st.markdown("### Recall por grupo")
            chart_df = df_groups.set_index("grupo")[["recall"]]
            st.bar_chart(chart_df)

        if "specificity" in df_groups.columns and "grupo" in df_groups.columns:
            st.markdown("### Specificity por grupo")
            chart_df = df_groups.set_index("grupo")[["specificity"]]
            st.bar_chart(chart_df)

st.divider()

with st.expander("Observações metodológicas"):
    st.markdown(
        """
- **Métricas do modelo salvo**: referem-se ao modelo final armazenado no arquivo `.pkl`.
- **Métricas durante a busca**: referem-se ao desempenho observado na fase em que o algoritmo genético testava combinações de hiperparâmetros.
- **Matriz de confusão**: mostra os acertos e erros do modelo no conjunto de teste usado pela função `load_data()`.
- **Equidade por grupo**: mostra se o desempenho varia entre faixas de raça, o que ajuda a identificar possíveis desigualdades no comportamento do modelo.
"""
    )