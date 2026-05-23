import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.explain.shap_explainer import get_shap_values
from src.optimize_ga import load_data
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


with st.sidebar:
    st.title("Sistema IA")
    st.subheader("Proteção da Mulher")

    model_label = st.selectbox(
        "Modelo preditivo",
        list(model_names.keys()),
        key="sidebar_modelo"
    )

    st.caption(f"Arquivo: {model_names[model_label]}")

    st.divider()

    page = st.radio(
        "Navegação",
        [
            "Simulação de Risco",
            "Assistente Inteligente",
            "Detalhes do Modelo",
            "Avaliação e Equidade",
        ],
        key="sidebar_pagina"
    )

    st.divider()

    st.markdown("""
**Tecnologias utilizadas**

- Random Forest  
- SHAP  
- LangGraph  
- RAG  
- Guardrails  
- Streamlit  
- LGPD  
""")


model_file = model_names[model_label]
artifact_path = MODELS_DIR / model_file
artifact = joblib.load(artifact_path)

model = artifact["model"]
expected_features = artifact["features"]


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
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
        "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
        "RO", "RR", "RS", "SC", "SE", "SP", "TO",
    ],
    "DIA_SEMANA_OCOR": ["DOM", "QUA", "QUI", "SAB", "SEG", "SEX", "TER"],
}

BINARY_FEATURES = [
    "AG_AMEACA",
    "AG_ENFOR",
    "AUTOR_ALCO",
    "VIOL_PSICO",
    "VIOL_FISIC",
    "VIOL_FINAN",
    "VIOL_SEXU",
    "REL_PARCEIRO_INTIMO",
    "REL_FAMILIAR",
    "REL_CONHECIDO",
    "REL_INSTITUCIONAL",
    "REL_OUTROS_FLAG",
    "DEF_TRANS",
    "CIRC_LESAO_FAMILIA",
]


FRIENDLY_NAMES = {
    "AG_AMEACA": "Ameaça",
    "AG_ENFOR": "Enforcamento",
    "AUTOR_ALCO": "Uso de álcool pelo agressor",
    "VIOL_PSICO": "Violência psicológica",
    "VIOL_FISIC": "Violência física",
    "VIOL_FINAN": "Violência financeira/patrimonial",
    "VIOL_SEXU": "Violência sexual",
    "REL_PARCEIRO_INTIMO": "Autor é parceiro íntimo",
    "REL_FAMILIAR": "Autor é familiar",
    "REL_CONHECIDO": "Autor é conhecido",
    "REL_INSTITUCIONAL": "Relação institucional",
    "REL_OUTROS_FLAG": "Outra relação",
    "DEF_TRANS": "Deficiência/transtorno",
    "CIRC_LESAO_FAMILIA": "Lesão em contexto familiar",
    "SIT_CONJUG": "Situação conjugal",
    "ESCOLARIDADE": "Escolaridade",
    "AUTOR_SEXO": "Sexo do autor",
    "CICL_VID": "Ciclo de vida",
    "SG_UF": "UF",
    "DIA_SEMANA_OCOR": "Dia da semana",
}


def build_prediction_row(expected_features, binary_vals, categorical_vals):
    row = {f: 0 for f in expected_features}

    for feat in BINARY_FEATURES:
        if feat in row:
            row[feat] = 1 if binary_vals.get(feat) else 0

    for feat, options in CATEGORICAL_OPTIONS.items():
        if feat in row and feat in categorical_vals:
            row[feat] = options.index(categorical_vals[feat])

    return row


if page == "Simulação de Risco":

    st.title("Simulação de Risco Individual")

    st.markdown("""
Esta área estima a probabilidade de recorrência de violência contra a mulher
com base nas **mesmas variáveis usadas no treinamento do modelo**.

A ideia é permitir uma simulação mais fiel ao dataset da Fase 2, mantendo uma
interface compreensível para pessoas não técnicas.
""")

    st.info(
        "Preencha apenas as informações conhecidas. Campos não marcados serão interpretados como ausência daquela informação."
    )

    st.warning(
        "A predição é apoio à decisão. Não substitui análise profissional, jurídica, policial, psicológica ou médica."
    )

    with st.expander("Como interpretar esta simulação?", expanded=True):
        st.markdown("""
- **Risco baixo:** menor probabilidade estimada de recorrência segundo o modelo.
- **Risco moderado:** sinais relevantes que recomendam avaliação profissional.
- **Risco alto:** combinação de fatores associada a maior probabilidade de recorrência.

O resultado vem do modelo Random Forest treinado na Fase 2.  
A explicação dos fatores é feita com SHAP.
""")

    st.divider()

    with st.form("form_simulacao_dataset_completo"):

        st.subheader("1. Meio de agressão identificado")

        ag1, ag2, ag3 = st.columns(3)

        with ag1:
            ag_ameaca = st.checkbox("Ameaça", key="ag_ameaca")
            ag_corte = st.checkbox("Objeto cortante/perfurante", key="ag_corte")
            ag_enfor = st.checkbox("Enforcamento", key="ag_enfor")

        with ag2:
            ag_enven = st.checkbox("Envenenamento", key="ag_enven")
            ag_fogo = st.checkbox("Arma de fogo", key="ag_fogo")
            ag_forca = st.checkbox("Força corporal/espancamento", key="ag_forca")

        with ag3:
            ag_objeto = st.checkbox("Objeto contundente", key="ag_objeto")
            ag_outros = st.checkbox("Outros meios", key="ag_outros")
            ag_quente = st.checkbox("Substância/objeto quente", key="ag_quente")

        st.divider()

        st.subheader("2. Tipo de violência registrada")

        v1, v2, v3 = st.columns(3)

        with v1:
            viol_finan = st.checkbox("Violência financeira/patrimonial", key="viol_finan")
            viol_fisic = st.checkbox("Violência física", key="viol_fisic")
            viol_infan = st.checkbox("Violência contra criança/adolescente", key="viol_infan")

        with v2:
            viol_legal = st.checkbox("Violência/intervenção legal", key="viol_legal")
            viol_negli = st.checkbox("Negligência/abandono", key="viol_negli")
            viol_outr = st.checkbox("Outras violências", key="viol_outr")

        with v3:
            viol_psico = st.checkbox("Violência psicológica/moral", key="viol_psico")
            viol_sexu = st.checkbox("Violência sexual", key="viol_sexu")
            viol_tort = st.checkbox("Tortura", key="viol_tort")
            viol_traf = st.checkbox("Tráfico de pessoas", key="viol_traf")

        st.divider()

        st.subheader("3. Contexto do agressor e da ocorrência")

        c1, c2, c3 = st.columns(3)

        with c1:
            autor_alco = st.checkbox("Uso de álcool pelo agressor", key="autor_alco")

        with c2:
            def_trans = st.checkbox("Deficiência/transtorno relacionado", key="def_trans")

        with c3:
            circ_lesao = st.checkbox("Lesão em contexto familiar", key="circ_lesao")

        st.divider()

        st.subheader("4. Relação entre vítima e autor")

        r1, r2, r3 = st.columns(3)

        with r1:
            rel_parceiro = st.checkbox("Parceiro íntimo ou ex-parceiro", key="rel_parceiro")
            rel_familiar = st.checkbox("Familiar", key="rel_familiar")

        with r2:
            rel_conhecido = st.checkbox("Conhecido", key="rel_conhecido")
            rel_institucional = st.checkbox("Relação institucional", key="rel_institucional")

        with r3:
            rel_outros = st.checkbox("Outros vínculos", key="rel_outros")

        st.divider()

        st.subheader("5. Informações cadastrais e temporais")

        d1, d2, d3 = st.columns(3)

        with d1:
            autor_sexo = st.selectbox(
                "Sexo do autor",
                CATEGORICAL_OPTIONS["AUTOR_SEXO"],
                key="autor_sexo_dataset"
            )

            sit_conjug = st.selectbox(
                "Situação conjugal",
                CATEGORICAL_OPTIONS["SIT_CONJUG"],
                key="sit_conjug_dataset"
            )

            escolaridade = st.selectbox(
                "Escolaridade",
                CATEGORICAL_OPTIONS["ESCOLARIDADE"],
                key="escolaridade_dataset"
            )

        with d2:
            cicl_vid = st.selectbox(
                "Ciclo de vida",
                CATEGORICAL_OPTIONS["CICL_VID"],
                key="cicl_vid_dataset"
            )

            sg_uf = st.selectbox(
                "UF",
                CATEGORICAL_OPTIONS["SG_UF"],
                index=CATEGORICAL_OPTIONS["SG_UF"].index("DF") if "DF" in CATEGORICAL_OPTIONS["SG_UF"] else 0,
                key="sg_uf_dataset"
            )

            dia_semana = st.selectbox(
                "Dia da semana da ocorrência",
                CATEGORICAL_OPTIONS["DIA_SEMANA_OCOR"],
                key="dia_semana_dataset"
            )

        with d3:
            cs_raca = st.selectbox(
                "Raça/cor",
                [
                    "Ignorado",
                    "Branca",
                    "Preta",
                    "Parda",
                    "Amarela",
                    "Indígena",
                ],
                key="cs_raca_dataset"
            )

            ident_gen = st.selectbox(
                "Identidade de gênero",
                [
                    "Ignorado",
                    "Mulher cis",
                    "Mulher trans",
                    "Homem cis",
                    "Homem trans",
                    "Não binário",
                ],
                key="ident_gen_dataset"
            )

            orient_sex = st.selectbox(
                "Orientação sexual",
                [
                    "Ignorado",
                    "Heterossexual",
                    "Homossexual",
                    "Bissexual",
                    "Outra",
                ],
                key="orient_sex_dataset"
            )

            mes_ocor = st.selectbox(
                "Mês da ocorrência",
                list(range(1, 13)),
                key="mes_ocor_dataset"
            )

        submitted = st.form_submit_button(
            "Avaliar risco com variáveis do dataset",
            type="primary",
            use_container_width=True
        )

    if submitted:

        binary_vals = {
            "AG_AMEACA": ag_ameaca,
            "AG_CORTE": ag_corte,
            "AG_ENFOR": ag_enfor,
            "AG_ENVEN": ag_enven,
            "AG_FOGO": ag_fogo,
            "AG_FORCA": ag_forca,
            "AG_OBJETO": ag_objeto,
            "AG_OUTROS": ag_outros,
            "AG_QUENTE": ag_quente,
            "AUTOR_ALCO": autor_alco,
            "DEF_TRANS": def_trans,
            "VIOL_FINAN": viol_finan,
            "VIOL_FISIC": viol_fisic,
            "VIOL_INFAN": viol_infan,
            "VIOL_LEGAL": viol_legal,
            "VIOL_NEGLI": viol_negli,
            "VIOL_OUTR": viol_outr,
            "VIOL_PSICO": viol_psico,
            "VIOL_SEXU": viol_sexu,
            "VIOL_TORT": viol_tort,
            "VIOL_TRAF": viol_traf,
            "REL_PARCEIRO_INTIMO": rel_parceiro,
            "REL_FAMILIAR": rel_familiar,
            "REL_CONHECIDO": rel_conhecido,
            "REL_INSTITUCIONAL": rel_institucional,
            "REL_OUTROS_FLAG": rel_outros,
            "CIRC_LESAO_FAMILIA": circ_lesao,
        }

        categorical_vals = {
            "AUTOR_SEXO": autor_sexo,
            "CICL_VID": cicl_vid,
            "SG_UF": sg_uf,
            "SIT_CONJUG": sit_conjug,
            "DIA_SEMANA_OCOR": dia_semana,
            "ESCOLARIDADE": escolaridade,
        }

        row = build_prediction_row(
            expected_features,
            binary_vals,
            categorical_vals
        )

        # Variáveis categóricas que foram codificadas no treinamento,
        # mas ainda não possuem mapeamento completo no app.
        if "CS_RACA" in row:
            row["CS_RACA"] = [
                "Ignorado",
                "Branca",
                "Preta",
                "Parda",
                "Amarela",
                "Indígena",
            ].index(cs_raca)

        if "IDENT_GEN" in row:
            row["IDENT_GEN"] = [
                "Ignorado",
                "Mulher cis",
                "Mulher trans",
                "Homem cis",
                "Homem trans",
                "Não binário",
            ].index(ident_gen)

        if "ORIENT_SEX" in row:
            row["ORIENT_SEX"] = [
                "Ignorado",
                "Heterossexual",
                "Homossexual",
                "Bissexual",
                "Outra",
            ].index(orient_sex)

        if "MES_OCOR" in row:
            row["MES_OCOR"] = int(mes_ocor)

        data = pd.DataFrame([row], columns=expected_features)
        prob = model.predict_proba(data)[0][1]

        st.divider()
        st.subheader("Resultado da avaliação")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("Probabilidade estimada", f"{prob:.0%}")

        with col_r3:
            st.metric("Modelo utilizado", model_label)

        if prob >= 0.7:
            risk_label = "Alto"
            with col_r2:
                st.metric("Classificação", "RISCO ALTO")
            st.error(
                "O modelo estimou risco alto de recorrência. Recomenda-se atenção prioritária e avaliação especializada."
            )

        elif prob >= 0.4:
            risk_label = "Moderado"
            with col_r2:
                st.metric("Classificação", "RISCO MODERADO")
            st.warning(
                "O modelo estimou risco moderado. Recomenda-se acolhimento e avaliação profissional."
            )

        else:
            risk_label = "Baixo"
            with col_r2:
                st.metric("Classificação", "RISCO BAIXO")
            st.success(
                "O modelo estimou risco baixo com base nas variáveis preenchidas."
            )

        st.divider()

        st.subheader("Variáveis enviadas ao modelo")

        with st.expander("Ver vetor usado na predição"):
            st.dataframe(data, use_container_width=True)

        st.subheader("Por que o sistema chegou a esse resultado?")

        shap_values = get_shap_values(model, data)

        sv = np.asarray(shap_values)
        sv_row = sv[0] if sv.ndim > 1 else sv
        sv_row = np.asarray(sv_row).reshape(-1)

        top_idx = [
            int(i) for i in np.argsort(np.abs(sv_row))[::-1]
            if int(i) < len(expected_features)
        ][:8]

        factors = []

        for i in top_idx:
            feature = expected_features[i]
            impact = float(np.asarray(sv_row[i]).item())

            factors.append({
                "Fator": FRIENDLY_NAMES.get(feature, feature),
                "Variável técnica": feature,
                "Valor enviado": data.iloc[0][feature],
                "Impacto SHAP": round(impact, 4),
                "Interpretação": (
                    "Aumentou o risco estimado"
                    if impact > 0
                    else "Reduziu o risco estimado"
                )
            })

        st.dataframe(
            pd.DataFrame(factors),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Observação metodológica")

        st.markdown(f"""
**Classificação:** {risk_label}  
**Probabilidade estimada:** {prob:.0%}

Esta versão utiliza todas as variáveis esperadas pelo modelo salvo no artefato `.pkl`.
Isso reduz inconsistências entre a interface e o dataset original da Fase 2.

Caso o resultado ainda pareça contraintuitivo, a causa provável está no próprio
treinamento do modelo, distribuição do dataset, desbalanceamento das classes
ou codificação das variáveis durante a preparação dos dados.
""")
        
elif page == "Assistente Inteligente":

    st.title("Assistente Inteligente para Proteção da Mulher")

    st.markdown("""
Plataforma de apoio à triagem, acolhimento e encaminhamento de casos
relacionados à proteção da mulher, integrando LangGraph, RAG, protocolos,
legislação, guardrails, explainability e auditoria.
""")

    st.warning(
        "Este sistema é uma ferramenta de apoio. Não substitui atendimento "
        "profissional, jurídico, médico, policial, psicológico ou de emergência."
    )

    st.divider()

    col_intro1, col_intro2, col_intro3 = st.columns(3)

    with col_intro1:
        st.metric("Arquitetura", "LangGraph")

    with col_intro2:
        st.metric("Base de conhecimento", "RAG")

    with col_intro3:
        st.metric("Segurança", "Guardrails")

    st.divider()

    st.subheader("Entrada do caso")

    example = st.selectbox(
        "Escolha um exemplo ou escreva seu próprio relato:",
        [
            "",
            "Tenho medo do meu parceiro. Ele me ameaça e controla minhas consultas.",
            "Estou grávida e tive sangramento hoje pela manhã.",
            "Estou muito ansiosa, chorando muito e sem conseguir dormir.",
            "Quero saber como prevenir situações de violência e onde buscar orientação.",
        ],
        key="assistente_exemplos"
    )

    user_input = st.text_area(
        "Relato",
        value=example if example else "",
        height=180,
        placeholder="Descreva o caso a ser analisado...",
        key="assistente_input"
    )

    analyze = st.button(
        "Analisar caso com IA",
        type="primary",
        use_container_width=True,
        key="assistente_botao"
    )

    if analyze:

        if not user_input.strip():
            st.warning("Digite um relato para análise.")
            st.stop()

        with st.spinner(
            "Executando fluxo LangGraph, consulta RAG e validação de segurança..."
        ):
            graph = build_graph()
            result = graph.invoke({"user_input": user_input})

        st.success("Análise concluída.")

        st.divider()

        st.subheader("Resumo da análise")

        flow = result.get("flow_type", "não identificado")
        risk = result.get("risk_level", "não informado")
        validation = result.get("audit_log", {}).get(
            "safety_validation",
            "não informado"
        )
        used_rag = result.get("audit_log", {}).get("used_rag", False)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Fluxo", flow)

        with col2:
            st.metric("Risco", risk.upper())

        with col3:
            st.metric("Validação", validation.upper())

        with col4:
            st.metric("RAG", "SIM" if used_rag else "NÃO")

        if risk == "crítico":
            st.error(
                "Risco crítico identificado. Recomenda-se encaminhamento imediato para atendimento especializado ou emergência."
            )
        elif risk == "alto":
            st.error(
                "Risco alto identificado. Recomenda-se atenção prioritária e encaminhamento para a rede especializada."
            )
        elif risk == "moderado":
            st.warning(
                "Risco moderado identificado. Recomenda-se acolhimento e avaliação profissional."
            )
        else:
            st.success(
                "Risco baixo identificado com base no relato informado."
            )

        st.divider()

        st.subheader("Fatores considerados")

        factors = result.get("risk_factors", [])

        if factors:
            for factor in factors:
                st.write(f"• {factor}")
        else:
            st.write("• Nenhum fator crítico explícito identificado.")

        st.divider()

        tab_resp, tab_context, tab_audit, tab_arch = st.tabs(
            [
                "Resposta segura",
                "Contexto recuperado",
                "Auditoria",
                "Fluxo técnico",
            ]
        )

        with tab_resp:
            st.markdown(result.get("response", ""))

        with tab_context:
            st.markdown(
                result.get(
                    "protocol_context",
                    "Nenhum contexto recuperado pelo RAG."
                )
            )

        with tab_audit:
            st.json(result.get("audit_log", {}))

        with tab_arch:
            st.markdown("""
**Fluxo LangGraph executado:**

1. Entrada do relato  
2. Classificação do fluxo  
3. Avaliação do risco  
4. Recuperação de contexto via RAG  
5. Definição de ações recomendadas  
6. Geração de explicação  
7. Registro de auditoria  
8. Validação por guardrails  
9. Resposta segura ao usuário  

**Componentes utilizados:**

- `src/graphs/womens_safety_graph.py`
- `src/rag/retriever.py`
- `src/guardrails/safety_guardrails.py`
- `data/protocols/`
""")


elif page == "Detalhes do Modelo":

    st.title(f"Detalhes do Modelo — {model_label}")

    tab_params, tab_metrics = st.tabs(["Parâmetros", "Métricas"])

    with tab_params:
        st.subheader("Hiperparâmetros")
        params = artifact.get("params", {})
        if isinstance(params, str):
            st.code(params)
        else:
            st.json(params)

    with tab_metrics:
        st.subheader("Métricas do modelo salvo")
        st.json(artifact.get("metrics", {}))


elif page == "Avaliação e Equidade":

    st.title(f"Avaliação e Equidade — {model_label}")

    tab_cm, tab_equity = st.tabs(["Matriz de Confusão", "Equidade por Grupo"])

    with tab_cm:
        _, _, _, _, X_test, _, y_test = load_data()

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False)
        st.pyplot(fig)

    with tab_equity:
        group_metrics = artifact.get("group_metrics", {})

        if group_metrics:
            st.json(group_metrics)
        else:
            st.info("Nenhuma métrica de grupo encontrada.")