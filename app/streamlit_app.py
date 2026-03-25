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

BASE_PATH = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_PATH / "models"

pkl_files = sorted(MODELS_DIR.glob("random_forest_*.pkl"))

if not pkl_files:
    st.error("Nenhum modelo encontrado em models/")
    st.stop()

model_names = {
    f.stem.replace("random_forest_", "").replace("_", " ").title(): f.name
    for f in pkl_files
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Modelo")
    model_label = st.selectbox("Selecione o modelo", list(model_names.keys()))
    st.caption(f"Arquivo: {model_names[model_label]}")

    st.divider()

    st.header("Navegação")

    page = st.radio(
        "Seção",
        [
            "Simulação de Risco",
            "Detalhes do Modelo",
            "Avaliação e Equidade",
            "Comparação de Experimentos",
        ],
        label_visibility="collapsed",
    )

    with st.expander("Observações metodológicas"):
        st.markdown(
            """
- **Métricas do modelo salvo**: modelo final no `.pkl`.
- **Métricas durante a busca**: desempenho durante otimização com algoritmo genético.
- **Matriz de confusão**: acertos e erros no conjunto de teste.
- **Equidade por grupo**: variação de desempenho entre faixas de raça.
- **Comparação**: baseline vs modelos otimizados.
"""
        )

# ── Carregamento do modelo ────────────────────────────────────────────────────

model_file = model_names[model_label]
artifact_path = MODELS_DIR / model_file
artifact = joblib.load(artifact_path)

model = artifact["model"]
expected_features = artifact["features"]


# ── Funções auxiliares ────────────────────────────────────────────────────────


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


# ── Página: Simulação de Risco ────────────────────────────────────────────────

if page == "Simulação de Risco":
    st.title("Simulação de Risco Individual")
    st.markdown(
        f"Modelo ativo: **{model_label}** · "
        "Preencha os dados abaixo e clique em **Avaliar** para obter a predição."
    )

    with st.form("form_simulacao"):
        col_form1, col_form2 = st.columns(2)

        with col_form1:
            idade = st.slider("Idade da vítima", 10, 80, 30)
            relacao = st.selectbox(
                "Relação com agressor",
                ["parceiro_intimo", "familiar", "conhecido", "outros"],
            )

        with col_form2:
            viol_psico = st.checkbox("Violência psicológica")
            ameaca = st.checkbox("Ameaça")
            alcool = st.checkbox("Uso de álcool pelo agressor")

        submitted = st.form_submit_button("Avaliar risco individual", type="primary")

    if submitted:
        row = build_prediction_row(
            expected_features, idade, relacao, viol_psico, ameaca, alcool
        )
        data = pd.DataFrame([row], columns=expected_features)

        prob = model.predict_proba(data)[0][1]

        if prob >= 0.7:
            st.error(f"Probabilidade de recorrência: **{prob:.0%}** — Risco alto")
        elif prob >= 0.4:
            st.warning(f"Probabilidade de recorrência: **{prob:.0%}** — Risco moderado")
        else:
            st.success(f"Probabilidade de recorrência: **{prob:.0%}** — Risco baixo")

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

        st.subheader("Principais fatores (SHAP)")
        st.dataframe(pd.DataFrame(top_features), width="stretch")

        st.subheader("Explicação com IA")
        st.info("Explicação com IA desativada temporariamente para testes locais.")
        # explanation = explain_case(prob, top_features)
        # st.write(explanation)

# ── Página: Detalhes do Modelo ────────────────────────────────────────────────

elif page == "Detalhes do Modelo":
    st.title(f"Detalhes do Modelo — {model_label}")

    tab_params, tab_metrics = st.tabs(["Parâmetros", "Métricas"])

    with tab_params:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hiperparâmetros")
            params = artifact.get("params", {})
            if isinstance(params, str):
                st.code(params)
            else:
                st.json(params)

        with col2:
            st.subheader("Informações gerais")
            info = {
                "fitness_mode": artifact.get("fitness_mode", "não informado"),
                "fitness": artifact.get("fitness", "não informado"),
                "arquivo": model_file,
            }
            ga_config = artifact.get("ga_config")
            if ga_config:
                info["ga_config"] = ga_config
            st.json(info)

    with tab_metrics:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("Métricas do modelo salvo")
            st.json(artifact.get("metrics", {}))

        with col_m2:
            st.subheader("Métricas durante a busca")
            search_metrics = artifact.get("search_metrics")
            if search_metrics:
                st.json(search_metrics)
            else:
                st.info("Modelo baseline — sem busca por hiperparâmetros.")

# ── Página: Avaliação e Equidade ──────────────────────────────────────────────

elif page == "Avaliação e Equidade":
    st.title(f"Avaliação — {model_label}")

    tab_cm, tab_equity = st.tabs(["Matriz de Confusão", "Equidade por Grupo"])

    with tab_cm:
        _, _, _, _, X_test, _, y_test = load_data()

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig, ax = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, colorbar=False)
            st.pyplot(fig)

        with col_table:
            summary_df = pd.DataFrame(
                [
                    ["Verdadeiros Negativos (TN)", int(tn)],
                    ["Falsos Positivos (FP)", int(fp)],
                    ["Falsos Negativos (FN)", int(fn)],
                    ["Verdadeiros Positivos (TP)", int(tp)],
                ],
                columns=["Tipo", "Quantidade"],
            )
            st.dataframe(summary_df, width="stretch", hide_index=True)

    with tab_equity:
        group_metrics = artifact.get("group_metrics", {})

        if not group_metrics:
            st.warning("Nenhuma métrica de grupo foi encontrada no artifact salvo.")
        else:
            summary = group_metrics.get("_summary", {})
            groups_only = {k: v for k, v in group_metrics.items() if k != "_summary"}

            if summary:
                st.subheader("Resumo de equidade")
                st.json(summary)

            if groups_only:
                st.subheader("Métricas por raça")

                rows = []
                for group_name, values in groups_only.items():
                    row = {"grupo": group_name}
                    row.update(values)
                    rows.append(row)

                df_groups = pd.DataFrame(rows)

                preferred_columns = [
                    "grupo", "size", "recall", "specificity",
                    "precision", "f1", "tn", "fp", "fn", "tp",
                ]
                existing_columns = [c for c in preferred_columns if c in df_groups.columns]
                df_groups = df_groups[existing_columns]

                st.dataframe(df_groups, width="stretch", hide_index=True)

                col_r, col_s = st.columns(2)

                with col_r:
                    if "recall" in df_groups.columns and "grupo" in df_groups.columns:
                        st.markdown("#### Recall por grupo")
                        chart_df = df_groups.set_index("grupo")[["recall"]]
                        st.bar_chart(chart_df)

                with col_s:
                    if "specificity" in df_groups.columns and "grupo" in df_groups.columns:
                        st.markdown("#### Specificity por grupo")
                        chart_df = df_groups.set_index("grupo")[["specificity"]]
                        st.bar_chart(chart_df)

# ── Página: Comparação de Experimentos ────────────────────────────────────────

elif page == "Comparação de Experimentos":
    st.title("Comparação: Baseline vs Experimentos")

    comparison_path = BASE_PATH / "results" / "comparison.csv"

    if not comparison_path.exists():
        st.warning("Arquivo `results/comparison.csv` não encontrado.")
        st.stop()

    comparison_df = pd.read_csv(comparison_path)
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        if "recall" in comparison_df.columns and "experiment" in comparison_df.columns:
            st.markdown("### Recall por experimento")
            chart_df = comparison_df.set_index("experiment")[["recall"]]
            st.bar_chart(chart_df)

    with col_c2:
        if "recall_gap" in comparison_df.columns and "experiment" in comparison_df.columns:
            st.markdown("### Recall gap (desigualdade)")
            chart_df = comparison_df.set_index("experiment")[["recall_gap"]]
            st.bar_chart(chart_df)
