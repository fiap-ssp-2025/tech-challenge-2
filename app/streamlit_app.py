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
from src.graphs.womens_safety_graph import build_graph


st.set_page_config(
    page_title="Sistema IA - Proteção da Mulher",
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
    st.subheader("Modelo preditivo")
    model_label = st.selectbox("Selecione o modelo", list(model_names.keys()))
    st.caption(f"Arquivo: {model_names[model_label]}")

    st.divider()

    st.header("Navegação")

    page = st.radio(
        "Seção",
        [
            "Simulação de Risco",
            "Assistente Inteligente",
            "Detalhes do Modelo",
            "Avaliação e Equidade",
            "Comparação de Experimentos",
        ],
        label_visibility="collapsed",
    )

    with st.expander("Observações metodológicas"):
        st.markdown(
            """
- **Simulação de Risco**: usa modelo Random Forest treinado na Fase 2.
- **Assistente Inteligente**: usa LangGraph, RAG, protocolos e guardrails da Fase 3.
- **SHAP**: mostra fatores que mais influenciaram a predição.
- **Equidade**: avalia desempenho por grupos.
- **Auditoria**: registra fluxo, risco e validação de segurança.
"""
        )

# ── Carregamento do modelo ────────────────────────────────────────────────────

model_file = model_names[model_label]
artifact_path = MODELS_DIR / model_file
artifact = joblib.load(artifact_path)

model = artifact["model"]
expected_features = artifact["features"]


# ── Funções auxiliares ────────────────────────────────────────────────────────

CATEGORICAL_OPTIONS = {
    "SIT_CONJUG": ["Casado/ União", "Não se aplica", "Separado", "Solteiro", "Viúvo"],
    "ESCOLARIDADE": [
        "ENSINO_MEDIO_COMPLETO",
        "ENSINO_MEDIO_INCOMPLETO",
        "ENSINO_SUPERIOR",
        "FUNDAMENTAL_INCOMPLETO",
    ],
    "AUTOR_SEXO": ["Ambos", "Feminino", "Ignorado", "Masculino"],
    "CICL_VID": [
        "Adolescente",
        "Criança",
        "Ignorado",
        "Jovem",
        "Pessoa adulta",
        "Pessoa idosa",
    ],
    "SG_UF": [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
        "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
        "RS", "SC", "SE", "SP", "TO",
    ],
    "DIA_SEMANA_OCOR": ["DOM", "QUA", "QUI", "SAB", "SEG", "SEX", "TER"],
}

BINARY_FEATURES = [
    "AG_AMEACA", "AG_ENFOR", "AUTOR_ALCO",
    "VIOL_PSICO", "VIOL_FISIC", "VIOL_FINAN", "VIOL_SEXU",
    "REL_PARCEIRO_INTIMO", "REL_FAMILIAR", "REL_CONHECIDO",
    "REL_INSTITUCIONAL", "REL_OUTROS_FLAG",
    "DEF_TRANS", "CIRC_LESAO_FAMILIA",
]


def build_prediction_row(expected_features, binary_vals, categorical_vals):
    row = {f: 0 for f in expected_features}

    for feat in BINARY_FEATURES:
        if feat in row:
            row[feat] = 1 if binary_vals.get(feat) else 0

    for feat, options in CATEGORICAL_OPTIONS.items():
        if feat in row and feat in categorical_vals:
            row[feat] = options.index(categorical_vals[feat])

    return row


# ── Página: Simulação de Risco ────────────────────────────────────────────────

if page == "Simulação de Risco":
    st.title("Simulação de Risco Individual")
    st.markdown(
        f"Modelo ativo: **{model_label}** · "
        "Preencha os dados abaixo e clique em **Avaliar** para obter a predição."
    )

    with st.form("form_simulacao"):
        st.markdown("##### Relação com o agressor")
        rc1, rc2, rc3 = st.columns(3)
        rel_parceiro = rc1.checkbox("Parceiro íntimo")
        rel_familiar = rc1.checkbox("Familiar")
        rel_conhecido = rc2.checkbox("Conhecido")
        rel_institucional = rc2.checkbox("Institucional")
        rel_outros = rc3.checkbox("Outros")

        st.markdown("##### Tipos de violência")
        vc1, vc2, vc3, vc4 = st.columns(4)
        viol_psico = vc1.checkbox("Psicológica")
        viol_fisic = vc2.checkbox("Física")
        viol_finan = vc3.checkbox("Financeira")
        viol_sexu = vc4.checkbox("Sexual")

        st.markdown("##### Meios de agressão")
        ac1, ac2 = st.columns(2)
        ameaca = ac1.checkbox("Ameaça")
        enfor = ac2.checkbox("Enforcamento")

        st.markdown("##### Agressor e contexto")
        cc1, cc2, cc3, cc4 = st.columns(4)
        alcool = cc1.checkbox("Uso de álcool pelo agressor")
        def_trans = cc2.checkbox("Deficiência/transtorno")
        circ_lesao = cc3.checkbox("Lesão em contexto familiar")
        autor_sexo = cc4.selectbox("Sexo do autor", CATEGORICAL_OPTIONS["AUTOR_SEXO"], index=3)

        st.markdown("##### Dados da vítima")
        dc1, dc2, dc3, dc4 = st.columns(4)
        sit_conjug = dc1.selectbox("Situação conjugal", CATEGORICAL_OPTIONS["SIT_CONJUG"])
        escolaridade = dc2.selectbox("Escolaridade", CATEGORICAL_OPTIONS["ESCOLARIDADE"])
        cicl_vid = dc3.selectbox("Ciclo de vida", CATEGORICAL_OPTIONS["CICL_VID"], index=4)
        sg_uf = dc4.selectbox("UF", CATEGORICAL_OPTIONS["SG_UF"], index=25)

        st.markdown("##### Ocorrência")
        dia_semana = st.selectbox("Dia da semana", CATEGORICAL_OPTIONS["DIA_SEMANA_OCOR"])

        submitted = st.form_submit_button("Avaliar risco individual", type="primary")

    if submitted:
        binary_vals = {
            "AG_AMEACA": ameaca,
            "AG_ENFOR": enfor,
            "AUTOR_ALCO": alcool,
            "VIOL_PSICO": viol_psico,
            "VIOL_FISIC": viol_fisic,
            "VIOL_FINAN": viol_finan,
            "VIOL_SEXU": viol_sexu,
            "REL_PARCEIRO_INTIMO": rel_parceiro,
            "REL_FAMILIAR": rel_familiar,
            "REL_CONHECIDO": rel_conhecido,
            "REL_INSTITUCIONAL": rel_institucional,
            "REL_OUTROS_FLAG": rel_outros,
            "DEF_TRANS": def_trans,
            "CIRC_LESAO_FAMILIA": circ_lesao,
        }

        categorical_vals = {
            "SIT_CONJUG": sit_conjug,
            "ESCOLARIDADE": escolaridade,
            "AUTOR_SEXO": autor_sexo,
            "CICL_VID": cicl_vid,
            "SG_UF": sg_uf,
            "DIA_SEMANA_OCOR": dia_semana,
        }

        row = build_prediction_row(expected_features, binary_vals, categorical_vals)
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

        st.subheader("Principais fatores explicativos do modelo")
        st.dataframe(pd.DataFrame(top_features), width="stretch", hide_index=True)

        st.subheader("Orientação assistida por IA")

        try:
            import importlib
            import src.llm.llm_explainer as _llm_mod
            importlib.reload(_llm_mod)
            result = _llm_mod.explain_case(prob, top_features)

            if isinstance(result, tuple) and len(result) == 2:
                summary, details = result
            else:
                summary, details = str(result), None

            st.info(summary, icon="⚠️")

            if details:
                with st.expander("Mais informações sobre os fatores"):
                    st.markdown(details)

        except Exception as e:
            st.warning("Não foi possível gerar a explicação textual por IA.")
            st.exception(e)


# ── Página: Assistente Inteligente ───────────────────────────────────────────

elif page == "Assistente Inteligente":
    st.title("Assistente Inteligente para Proteção da Mulher")

    st.markdown(
        """
Este módulo representa a **Fase 3** do projeto, integrando LangGraph,
RAG, protocolos institucionais, legislação, guardrails, explainability
e auditoria para apoiar a triagem e o encaminhamento de casos envolvendo
violência contra a mulher.
"""
    )

    st.info(
        "O assistente não substitui profissionais de saúde, segurança pública, "
        "assistência social ou atendimento de emergência. Ele atua como apoio "
        "informativo e estruturado à decisão."
    )

    example = st.selectbox(
        "Escolha um exemplo ou escreva seu próprio relato:",
        [
            "",
            "Tenho medo do meu parceiro. Ele me ameaça e controla minhas consultas.",
            "Estou grávida e tive sangramento hoje pela manhã.",
            "Estou muito ansiosa, chorando muito e sem conseguir dormir.",
            "Quero saber como prevenir situações de violência e onde buscar orientação.",
        ],
    )

    default_text = example if example else ""

    user_input = st.text_area(
        "Relato do caso",
        value=default_text,
        height=180,
        placeholder="Descreva o caso a ser analisado..."
    )

    if st.button("Analisar caso com LangGraph", type="primary"):
        if not user_input.strip():
            st.warning("Digite um relato para análise.")
            st.stop()

        with st.spinner("Analisando caso com LangGraph e RAG..."):
            graph = build_graph()
            result = graph.invoke({"user_input": user_input})

        st.success("Análise concluída.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fluxo", result.get("flow_type", "não identificado"))

        with col2:
            risk = result.get("risk_level", "não informado")
            st.metric("Risco", risk.upper())

        with col3:
            validation = result.get("audit_log", {}).get("safety_validation", "não informado")
            st.metric("Validação", validation.upper())

        risk = result.get("risk_level", "")

        if risk == "crítico":
            st.error("Risco crítico identificado. Recomenda-se encaminhamento imediato para atendimento especializado ou emergência, conforme o contexto.")
        elif risk == "alto":
            st.error("Risco alto identificado. Recomenda-se atenção prioritária e encaminhamento para rede especializada.")
        elif risk == "moderado":
            st.warning("Risco moderado identificado. Recomenda-se acolhimento e avaliação profissional.")
        else:
            st.success("Risco baixo identificado com base no relato informado.")

        st.subheader("Fatores considerados")
        factors = result.get("risk_factors", [])

        if factors:
            for factor in factors:
                st.write(f"- {factor}")
        else:
            st.write("- Nenhum fator crítico explícito identificado.")

        tab_resp, tab_context, tab_audit = st.tabs(
            ["Resposta Segura", "Contexto RAG", "Auditoria"]
        )

        with tab_resp:
            st.markdown(result.get("response", ""))

        with tab_context:
            st.markdown(result.get("protocol_context", "Nenhum contexto recuperado."))

        with tab_audit:
            st.json(result.get("audit_log", {}))


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
            st.warning("Nenhuma métrica de grupo foi encontrada no artefato salvo.")
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

                existing_columns = [
                    c for c in preferred_columns if c in df_groups.columns
                ]

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