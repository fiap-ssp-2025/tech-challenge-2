import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.load_model import load_model
from src.explain.shap_explainer import get_shap_values
from src.llm.llm_explainer import explain_case

st.set_page_config(page_title="Violência Doméstica — DF", layout="wide")
st.title("Sistema de Avaliação de Risco de Violência Doméstica")

# ── Abas principais ─────────────────────────────────────────────────────────
tab_risco, tab_cronograma = st.tabs(
    ["🔍 Avaliação de Risco Individual", "📅 Cronograma Semanal de Visitas"]
)

# ════════════════════════════════════════════════════════════════════════════
#  ABA 1 — Avaliação de Risco Individual (código original)
# ════════════════════════════════════════════════════════════════════════════
with tab_risco:
    artifact = load_model()
    model = artifact["model"]
    expected_features = artifact["features"]

    idade = st.slider("Idade da vítima", 10, 80, 30)

    relacao = st.selectbox(
        "Relação com agressor",
        ["parceiro_intimo", "familiar", "conhecido", "outros"],
    )

    viol_psico = st.checkbox("Violência psicológica")
    ameaca = st.checkbox("Ameaça")
    alcool = st.checkbox("Uso de álcool pelo agressor")

    def build_prediction_row(
        expected_features, idade, relacao, viol_psico, ameaca, alcool
    ):
        """Monta uma linha com todas as features esperadas pelo modelo."""
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

    if st.button("Avaliar risco"):
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
            int(i)
            for i in np.argsort(np.abs(sv_row))[::-1]
            if int(i) < n_features
        ][:3]
        top_features = [
            (expected_features[i], float(np.asarray(sv_row[i]).item()))
            for i in top_idx
        ]

        st.write("Principais fatores:", top_features)

        st.subheader("Explicação com IA")
        try:
            explanation = explain_case(prob, top_features)
            st.write(explanation)
        except Exception as e:
            st.error(
                "Não foi possível gerar a explicação com IA. "
                "Verifique a chave OPENAI_API_KEY no .env e sua conexão."
            )
            st.caption(str(e))

# ════════════════════════════════════════════════════════════════════════════
#  ABA 2 — Cronograma Semanal de Visitas (tabela + mapa)
# ════════════════════════════════════════════════════════════════════════════
with tab_cronograma:
    import folium
    from streamlit_folium import st_folium

    PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cronograma_semanal.parquet"

    if not PARQUET_PATH.exists():
        st.warning(
            "Arquivo de cronograma não encontrado. Execute o notebook "
            "`ga_vrptw_cronograma.ipynb` para gerar o cronograma."
        )
    else:
        df_cron = pd.read_parquet(PARQUET_PATH)

        st.header("Cronograma Semanal — Visitas a Vítimas Priorizadas")

        # ── Métricas gerais ──────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de visitas", len(df_cron))
        m2.metric("Equipes", df_cron["equipe"].nunique())
        m3.metric("Dias", df_cron["dia"].nunique())
        m4.metric(
            "Risco médio",
            f"{df_cron['risco_prob'].mean():.2%}",
        )

        # ── Filtros na sidebar ───────────────────────────────────────────
        st.subheader("Filtros")
        fc1, fc2 = st.columns(2)

        with fc1:
            equipes_sel = st.multiselect(
                "Equipe(s)",
                options=sorted(df_cron["equipe"].unique()),
                default=sorted(df_cron["equipe"].unique()),
            )
        with fc2:
            dias_sel = st.multiselect(
                "Dia(s) da semana",
                options=sorted(df_cron["dia"].unique()),
                default=sorted(df_cron["dia"].unique()),
                format_func=lambda d: df_cron.loc[
                    df_cron["dia"] == d, "dia_semana"
                ].iloc[0],
            )

        df_filt = df_cron[
            df_cron["equipe"].isin(equipes_sel) & df_cron["dia"].isin(dias_sel)
        ]

        # ── Tabela estilizada ────────────────────────────────────────────
        st.subheader(f"Tabela de Visitas ({len(df_filt)} registros)")

        df_display = df_filt[
            [
                "equipe",
                "orgao_base",
                "dia_semana",
                "ordem_visita",
                "id_vitima",
                "risco_prob",
                "hora_chegada",
                "hora_saida",
                "tempo_deslocamento_min",
            ]
        ].copy()
        df_display.rename(
            columns={
                "equipe": "Equipe",
                "orgao_base": "Órgão Base",
                "dia_semana": "Dia",
                "ordem_visita": "Ordem",
                "id_vitima": "ID Vítima",
                "risco_prob": "Risco",
                "hora_chegada": "Chegada",
                "hora_saida": "Saída",
                "tempo_deslocamento_min": "Desloc. (min)",
            },
            inplace=True,
        )

        st.dataframe(
            df_display.style.background_gradient(
                subset=["Risco"], cmap="YlOrRd", vmin=0.90, vmax=1.0
            ).format({"Risco": "{:.4f}", "Desloc. (min)": "{:.1f}"}),
            use_container_width=True,
            height=450,
        )

        # ── Mapa interativo ──────────────────────────────────────────────
        st.subheader("Mapa de Rotas")

        TEAM_COLORS = [
            "#1f77b4",  # equipe 1 – azul
            "#ff7f0e",  # equipe 2 – laranja
            "#2ca02c",  # equipe 3 – verde
            "#d62728",  # equipe 4 – vermelho
            "#9467bd",  # equipe 5 – roxo
            "#8c564b",  # equipe 6 – marrom
            "#e377c2",  # equipe 7 – rosa
            "#17becf",  # equipe 8 – ciano
        ]

        center_lat = df_filt["lat"].mean() if len(df_filt) else -15.79
        center_lon = df_filt["lon"].mean() if len(df_filt) else -47.88

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="CartoDB positron",
        )

        # Marcadores das bases (ícone diferenciado)
        bases_shown = set()
        for _, row in df_filt.iterrows():
            eq = int(row["equipe"])
            if eq not in bases_shown:
                bases_shown.add(eq)
                folium.Marker(
                    location=[row["base_lat"], row["base_lon"]],
                    popup=f"<b>Base Equipe {eq}</b><br>{row['orgao_base']}",
                    tooltip=f"Base {eq}: {row['orgao_base']}",
                    icon=folium.Icon(
                        color="black",
                        icon_color=TEAM_COLORS[eq - 1],
                        icon="home",
                        prefix="fa",
                    ),
                ).add_to(m)

        # Rotas diárias (polylines) e marcadores de vítimas
        for (eq, dia), grp in df_filt.groupby(["equipe", "dia"]):
            eq = int(eq)
            color = TEAM_COLORS[eq - 1]
            grp_sorted = grp.sort_values("ordem_visita")

            # Polyline: base → visitas (em ordem)
            base_row = grp_sorted.iloc[0]
            coords = [[base_row["base_lat"], base_row["base_lon"]]]
            for _, v in grp_sorted.iterrows():
                coords.append([v["lat"], v["lon"]])

            dia_nome = grp_sorted["dia_semana"].iloc[0]
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=2,
                opacity=0.6,
                tooltip=f"Equipe {eq} — {dia_nome}",
            ).add_to(m)

            # Marcadores de vítimas
            for _, v in grp_sorted.iterrows():
                folium.CircleMarker(
                    location=[v["lat"], v["lon"]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    tooltip=(
                        f"<b>{v['id_vitima']}</b><br>"
                        f"Equipe {eq} | {v['dia_semana']}<br>"
                        f"Ordem: {int(v['ordem_visita'])}<br>"
                        f"Risco: {v['risco_prob']:.4f}<br>"
                        f"Chegada: {v['hora_chegada']} — Saída: {v['hora_saida']}"
                    ),
                ).add_to(m)

        # Legenda de equipes
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
             background: white; padding: 10px 14px; border-radius: 8px;
             border: 1px solid #ccc; font-size: 13px; line-height: 1.6;">
        <b>Equipes</b><br>
        """
        for i, base in enumerate(
            sorted(df_cron[["equipe", "orgao_base"]].drop_duplicates().values.tolist())
        ):
            legend_html += (
                f'<span style="color:{TEAM_COLORS[i]};font-size:16px;">&#9679;</span> '
                f"Eq {base[0]}: {base[1]}<br>"
            )
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, use_container_width=True, height=600)
