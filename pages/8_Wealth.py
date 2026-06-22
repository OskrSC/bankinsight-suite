"""
Módulo 8 — Wealth Management
KNN para recomendación de estrategia de inversión · perfil de riesgo ·
asset allocation sugerido · comparativa de portafolios
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import (
    gen_wealth_data, build_wealth_model,
    hex_to_rgba, base_layout, COLORS,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #7c3aed;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
</style>
""", unsafe_allow_html=True)

STRATEGY_COLORS = {
    "Conservadora": "#16a34a",
    "Balanceada":   "#0ea5e9",
    "Agresiva":     "#dc2626",
}
STRATEGY_MAP = {1: "Conservadora", 2: "Balanceada", 3: "Agresiva"}

# Asignación de activos sugerida por estrategia
ASSET_ALLOCATION = {
    "Conservadora": {"CETES/Deuda gob.": 55, "Bonos corporativos": 25,
                     "Renta variable": 10, "Efectivo": 10},
    "Balanceada":   {"CETES/Deuda gob.": 30, "Bonos corporativos": 25,
                     "Renta variable": 35, "Efectivo": 10},
    "Agresiva":     {"CETES/Deuda gob.": 10, "Bonos corporativos": 15,
                     "Renta variable": 65, "Efectivo": 10},
}
EXPECTED_RETURN = {"Conservadora": 0.065, "Balanceada": 0.095, "Agresiva": 0.135}
EXPECTED_VOL    = {"Conservadora": 0.04,  "Balanceada": 0.10,  "Agresiva": 0.20}

STRATEGY_DESC = {
    "Conservadora": "Prioriza preservación de capital. Ideal para horizontes cortos, "
                    "baja tolerancia al riesgo, o cercanía a la meta financiera.",
    "Balanceada":   "Equilibrio entre crecimiento y estabilidad. Adecuada para la "
                    "mayoría de inversionistas con horizonte de mediano-largo plazo.",
    "Agresiva":     "Maximiza crecimiento aceptando mayor volatilidad. Apropiada para "
                    "horizontes largos (10+ años) y alta tolerancia al riesgo.",
}

# ── Datos y modelo ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Entrenando recomendador KNN…")
def load():
    df = gen_wealth_data(2000)
    return df, *build_wealth_model(df)

df, model, scaler, feats, X_te, y_te, y_pred, acc = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 💼 Wealth Management")
st.markdown(
    "Recomendador de estrategia de inversión con **K-Nearest Neighbors (KNN)**. "
    "Clasifica el perfil del inversionista en Conservadora, Balanceada o Agresiva "
    "y sugiere una asignación de activos basada en el perfil más similar."
)
st.markdown("---")

strat_counts = df["strategy_label"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precisión del modelo", f"{acc:.1%}", "KNN (k=7)")
c2.metric("Perfiles analizados",  f"{len(df):,}")
c3.metric("Estrategia más común", strat_counts.idxmax(), f"{strat_counts.max()/len(df):.0%}")
c4.metric("Variables del perfil", str(len(feats)))

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Perfil del inversionista
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🧑‍💼 Perfil del Inversionista</div>', unsafe_allow_html=True)

with st.form("wealth_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.slider("Edad", 22, 70, 35)
        income = st.number_input("Ingreso mensual (MXN)", 10_000, 500_000, 45_000, step=1_000)
    with c2:
        risk_tol = st.select_slider(
            "Tolerancia al riesgo", options=[1, 2, 3],
            value=2, format_func=lambda x: {1: "Baja", 2: "Media", 3: "Alta"}[x],
        )
        horizon = st.slider("Horizonte de inversión (años)", 1, 30, 10)
    with c3:
        dependents = st.slider("Dependientes económicos", 0, 5, 1)
        debt_ratio = st.slider("Razón deuda/ingreso", 0.0, 0.60, 0.20, step=0.01)
        experience = st.select_slider(
            "Experiencia invirtiendo", options=[1, 2, 3],
            value=2, format_func=lambda x: {1: "Principiante", 2: "Intermedio", 3: "Avanzado"}[x],
        )
    submitted = st.form_submit_button(
        "🔍 Recomendar estrategia", use_container_width=True, type="primary"
    )

if submitted:
    X_in = scaler.transform([[age, income, risk_tol, horizon, dependents, debt_ratio, experience]])
    pred_strategy = int(model.predict(X_in)[0])
    probas = model.predict_proba(X_in)[0]
    strategy_label = STRATEGY_MAP[pred_strategy]
    color = STRATEGY_COLORS[strategy_label]

    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.markdown(f"""
        <div style='background:{hex_to_rgba(color,.10)};border:3px solid {color};
             border-radius:14px;padding:24px;text-align:center'>
          <div style='font-size:0.8rem;color:#64748b;font-weight:600'>ESTRATEGIA RECOMENDADA</div>
          <div style='font-size:1.9rem;font-weight:900;color:{color};margin:8px 0'>
            {strategy_label}
          </div>
        </div>""", unsafe_allow_html=True)
        st.metric("Retorno esperado anual", f"{EXPECTED_RETURN[strategy_label]:.1%}")
        st.metric("Volatilidad esperada",   f"{EXPECTED_VOL[strategy_label]:.1%}")
    with r2:
        classes = [STRATEGY_MAP[c] for c in model.classes_]
        fig_proba = go.Figure(go.Bar(
            x=classes, y=probas,
            marker_color=[STRATEGY_COLORS[c] for c in classes],
            text=[f"{p:.0%}" for p in probas], textposition="outside",
        ))
        fig_proba.update_layout(
            title="Probabilidad por estrategia (KNN)",
            yaxis_range=[0, 1.05], height=280,
            paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
            margin=dict(t=44, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_proba, use_container_width=True)
    with r3:
        alloc = ASSET_ALLOCATION[strategy_label]
        fig_alloc = go.Figure(go.Pie(
            labels=list(alloc.keys()), values=list(alloc.values()),
            marker_colors=["#0ea5e9", "#7c3aed", "#16a34a", "#94a3b8"],
            hole=0.42, textinfo="label+percent",
        ))
        fig_alloc.update_layout(
            title=f"Asignación de Activos Sugerida — {strategy_label}",
            height=300, paper_bgcolor="white",
            margin=dict(t=44, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">📋 <strong>{strategy_label}:</strong> {STRATEGY_DESC[strategy_label]}</div>
    """, unsafe_allow_html=True)

    # Vecinos más cercanos (explicabilidad de KNN)
    distances, indices = model.kneighbors(X_in, n_neighbors=7)
    neighbors_df = df.iloc[indices[0]][
        ["age", "income", "risk_tolerance", "horizon_years",
        "dependents", "debt_ratio", "invest_experience", "strategy_label"]
    ].copy()
    neighbors_df["Distancia"] = distances[0].round(3)
    neighbors_df = neighbors_df.rename(columns={
        "age": "Edad", "income": "Ingreso", "risk_tolerance": "Riesgo",
        "horizon_years": "Horizonte", "dependents": "Dependientes",
        "debt_ratio": "Deuda/Ing.", "invest_experience": "Experiencia",
        "strategy_label": "Estrategia",
    })
    with st.expander("🔍 Ver los 7 perfiles más similares (vecinos KNN)"):
        st.dataframe(
            neighbors_df.style.format({"Ingreso": "${:,.0f}", "Deuda/Ing.": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )
        st.markdown(
            '<div class="info-box">KNN clasifica buscando los k=7 perfiles más parecidos '
            'en el dataset histórico y asigna la estrategia mayoritaria entre ellos. '
            'Esta tabla muestra exactamente esos vecinos y su distancia euclidiana.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Comparativa de estrategias
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">⚖️ Comparativa de Estrategias</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Riesgo-Retorno", "🥧 Asignaciones", "📈 Simulación de crecimiento"])

with tab1:
    fig_rr = go.Figure()
    for strat in ["Conservadora", "Balanceada", "Agresiva"]:
        fig_rr.add_trace(go.Scatter(
            x=[EXPECTED_VOL[strat]], y=[EXPECTED_RETURN[strat]],
            mode="markers+text", text=[strat], textposition="top center",
            marker=dict(size=22, color=STRATEGY_COLORS[strat]),
            name=strat,
        ))
    fig_rr.update_layout(
        title="Mapa de Riesgo-Retorno por Estrategia",
        xaxis_title="Volatilidad esperada (riesgo)",
        yaxis_title="Retorno esperado anual",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        showlegend=False,
        **base_layout(),
    )
    st.plotly_chart(fig_rr, use_container_width=True)
    st.markdown('<div class="info-box">A mayor retorno esperado, mayor volatilidad asociada — es la relación fundamental riesgo-retorno en finanzas (no existe almuerzo gratis).</div>', unsafe_allow_html=True)

with tab2:
    alloc_cols = st.columns(3)
    for col, strat in zip(alloc_cols, ["Conservadora", "Balanceada", "Agresiva"]):
        alloc = ASSET_ALLOCATION[strat]
        with col:
            fig = go.Figure(go.Pie(
                labels=list(alloc.keys()), values=list(alloc.values()),
                marker_colors=["#0ea5e9", "#7c3aed", "#16a34a", "#94a3b8"],
                hole=0.42, textinfo="percent",
            ))
            fig.update_layout(
                title=strat, height=300, paper_bgcolor="white",
                margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.15, font=dict(size=9)),
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    sim_initial = st.number_input("Inversión inicial (MXN)", 1_000, 5_000_000, 100_000, step=10_000)
    sim_years    = st.slider("Horizonte de simulación (años)", 1, 30, 15, key="sim_years_wealth")

    fig_sim = go.Figure()
    np.random.seed(42)
    for strat in ["Conservadora", "Balanceada", "Agresiva"]:
        mu, sigma = EXPECTED_RETURN[strat], EXPECTED_VOL[strat]
        years_arr = np.arange(0, sim_years + 1)
        # Trayectoria determinística (retorno esperado)
        det_path = sim_initial * (1 + mu) ** years_arr
        fig_sim.add_trace(go.Scatter(
            x=years_arr, y=det_path, mode="lines",
            name=f"{strat} (esperado)",
            line=dict(color=STRATEGY_COLORS[strat], width=2.5),
        ))
        # Banda de incertidumbre ±1 sigma acumulada
        upper = sim_initial * (1 + mu + sigma) ** years_arr
        lower_base = max(1 + mu - sigma, 0.01)   # evita base negativa/cero en la potencia
        lower = sim_initial * lower_base ** years_arr
        fig_sim.add_trace(go.Scatter(
            x=np.concatenate([years_arr, years_arr[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(STRATEGY_COLORS[strat], 0.08),
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
    fig_sim.update_layout(
        title=f"Proyección de ${sim_initial:,.0f} por Estrategia ({sim_years} años)",
        xaxis_title="Año", yaxis_title="Valor del portafolio ($)",
        legend=dict(orientation="h", y=-0.18),
        **base_layout(),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    final_vals = {s: sim_initial * (1 + EXPECTED_RETURN[s]) ** sim_years
                 for s in ["Conservadora", "Balanceada", "Agresiva"]}
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Conservadora", f"${final_vals['Conservadora']:,.0f}")
    fc2.metric("Balanceada",   f"${final_vals['Balanceada']:,.0f}",
              f"+{(final_vals['Balanceada']/final_vals['Conservadora']-1):.0%}")
    fc3.metric("Agresiva",     f"${final_vals['Agresiva']:,.0f}",
              f"+{(final_vals['Agresiva']/final_vals['Conservadora']-1):.0%}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Explorador del dataset de perfiles
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🗂️ Explorador de Perfiles de Inversionistas</div>', unsafe_allow_html=True)

ca, cb = st.columns(2)
with ca:
    fig_dist = go.Figure(go.Bar(
        x=strat_counts.index, y=strat_counts.values,
        marker_color=[STRATEGY_COLORS[s] for s in strat_counts.index],
        text=strat_counts.values, textposition="outside",
    ))
    fig_dist.update_layout(
        title="Distribución de Estrategias en el Dataset",
        xaxis_title="Estrategia", yaxis_title="Inversionistas",
        **base_layout(),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
with cb:
    sample = df.sample(min(600, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="horizon_years", y="risk_tolerance", color="strategy_label",
        color_discrete_map=STRATEGY_COLORS, opacity=0.6,
        labels={"horizon_years": "Horizonte (años)", "risk_tolerance": "Tolerancia al riesgo"},
        title="Horizonte vs Tolerancia al Riesgo",
    )
    fig_scatter.update_layout(paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
                              margin=dict(t=44, b=44, l=60, r=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

with st.expander("📋 Ver muestra del dataset completo"):
    show = df.sample(min(200, len(df)), random_state=1)[
        ["age", "income", "risk_tolerance", "horizon_years",
        "dependents", "debt_ratio", "invest_experience", "strategy_label"]
    ].rename(columns={
        "age": "Edad", "income": "Ingreso", "risk_tolerance": "Riesgo",
        "horizon_years": "Horizonte", "dependents": "Dependientes",
        "debt_ratio": "Deuda/Ing.", "invest_experience": "Experiencia",
        "strategy_label": "Estrategia",
    })
    st.dataframe(
        show.style.format({"Ingreso": "${:,.0f}", "Deuda/Ing.": "{:.1%}"}),
        use_container_width=True, height=350,
    )
