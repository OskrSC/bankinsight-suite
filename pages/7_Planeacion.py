"""
Módulo 7 — Planeación Financiera
Calculadora de presupuesto mensual · metas de ahorro · proyección patrimonial por escenario
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import (
    EXPENSE_CATEGORIES, DEFAULT_EXPENSES, project_wealth,
    hex_to_rgba, base_layout, COLORS,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #d97706;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
.warn-box{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#78350f;margin:8px 0}
.success-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#14532d;margin:8px 0}
.danger-box{background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#7f1d1d;margin:8px 0}
</style>
""", unsafe_allow_html=True)

CAT_COLORS = {
    "Vivienda": "#0ea5e9", "Alimentación": "#16a34a", "Transporte": "#d97706",
    "Salud": "#dc2626", "Educación": "#7c3aed", "Entretenimiento": "#0891b2",
    "Ropa": "#db2777", "Ahorro forzoso": "#059669", "Otros": "#64748b",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📋 Planeación Financiera")
st.markdown(
    "Calculadora integral de **presupuesto mensual**, **metas de ahorro** y "
    "**proyección patrimonial** a largo plazo. Visualiza el impacto de tus decisiones "
    "financieras antes de tomarlas."
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Presupuesto mensual
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">💰 Presupuesto Mensual</div>', unsafe_allow_html=True)

income = st.number_input("Ingreso mensual neto (MXN)", 1_000, 500_000, 30_000, step=500)

st.markdown("**Ajusta tus gastos por categoría:**")
expense_cols = st.columns(3)
expenses = {}
for i, cat in enumerate(EXPENSE_CATEGORIES):
    col = expense_cols[i % 3]
    expenses[cat] = col.number_input(
        cat, 0, 100_000, DEFAULT_EXPENSES.get(cat, 0), step=100, key=f"exp_{cat}"
    )

total_expenses   = sum(expenses.values())
remaining_income = income - total_expenses
savings_rate     = remaining_income / income if income > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ingreso mensual",     f"${income:,.0f}")
c2.metric("Gastos totales",      f"${total_expenses:,.0f}", f"{total_expenses/income:.1%} del ingreso")
c3.metric("Disponible/Déficit",  f"${remaining_income:,.0f}",
          f"{savings_rate:+.1%}", delta_color="normal" if remaining_income >= 0 else "inverse")
c4.metric("Tasa de ahorro",      f"{max(savings_rate,0):.1%}",
          "Recomendado: ≥20%")

if remaining_income < 0:
    st.markdown(
        f'<div class="danger-box">🚨 <strong>Déficit de ${abs(remaining_income):,.0f}/mes.</strong> '
        f'Tus gastos superan tu ingreso. Revisa las categorías más altas para encontrar '
        f'oportunidades de recorte.</div>', unsafe_allow_html=True,
    )
elif savings_rate < 0.10:
    st.markdown(
        f'<div class="warn-box">⚠️ Tu tasa de ahorro ({savings_rate:.1%}) está por debajo del '
        f'10% recomendado mínimo. Considera reducir gastos discrecionales.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="success-box">✅ Tasa de ahorro saludable ({savings_rate:.1%}). '
        f'Estás en buen camino para construir patrimonio a largo plazo.</div>',
        unsafe_allow_html=True,
    )

ca, cb = st.columns(2)
with ca:
    exp_df = pd.DataFrame({"Categoría": list(expenses.keys()), "Monto": list(expenses.values())})
    exp_df = exp_df[exp_df["Monto"] > 0]
    fig_pie = go.Figure(go.Pie(
        labels=exp_df["Categoría"], values=exp_df["Monto"],
        marker_colors=[CAT_COLORS.get(c, COLORS["neutral"]) for c in exp_df["Categoría"]],
        hole=0.42, textinfo="label+percent",
        hovertemplate="%{label}: $%{value:,.0f}<extra></extra>",
    ))
    fig_pie.update_layout(
        title="Distribución del Gasto Mensual",
        height=380, paper_bgcolor="white",
        margin=dict(t=44, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)
with cb:
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=["Ingreso"], x=[income], orientation="h",
        marker_color=COLORS["success"], name="Ingreso",
        hovertemplate="$%{x:,.0f}<extra></extra>",
    ))
    cum = 0
    for cat, val in expenses.items():
        if val == 0:
            continue
        fig_bar.add_trace(go.Bar(
            y=["Gastos"], x=[val], orientation="h", name=cat,
            marker_color=CAT_COLORS.get(cat, COLORS["neutral"]),
            hovertemplate=f"{cat}: $%{{x:,.0f}}<extra></extra>",
            base=cum,
        ))
        cum += val
    fig_bar.update_layout(
        barmode="stack",
        title="Ingreso vs Gastos Acumulados",
        xaxis_title="Monto ($)",
        height=380, paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
        showlegend=False,
        margin=dict(t=44, b=44, l=80, r=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Metas de ahorro
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🎯 Meta de Ahorro</div>', unsafe_allow_html=True)

gc1, gc2, gc3 = st.columns(3)
with gc1:
    goal_name   = st.text_input("Nombre de la meta", "Enganche de casa")
    goal_amount = st.number_input("Monto objetivo (MXN)", 1_000, 10_000_000, 300_000, step=10_000)
with gc2:
    goal_initial = st.number_input("Ahorro inicial (MXN)", 0, 5_000_000, 20_000, step=1_000)
    goal_rate    = st.slider("Rendimiento anual esperado (%)", 0.0, 15.0, 8.0, step=0.5) / 100
with gc3:
    monthly_contribution = st.number_input(
        "Aportación mensual (MXN)", 0, 100_000,
        max(int(remaining_income * 0.5), 500), step=500,
        help="Por defecto se sugiere el 50% de tu disponible mensual",
    )

# Calcular meses necesarios (simulación mes a mes)
balance = goal_initial
months  = 0
monthly_rate = goal_rate / 12
history = [balance]
max_months = 600  # 50 años límite de seguridad
while balance < goal_amount and months < max_months:
    balance = balance * (1 + monthly_rate) + monthly_contribution
    months += 1
    history.append(balance)

years_needed  = months / 12
goal_progress = min(goal_initial / goal_amount, 1.0)

gm1, gm2, gm3, gm4 = st.columns(4)
gm1.metric("Meta", f"${goal_amount:,.0f}")
gm2.metric("Progreso actual", f"{goal_progress:.1%}", f"${goal_initial:,.0f} ahorrado")
if months >= max_months:
    gm3.metric("Tiempo estimado", "No alcanzable", "Aumenta tu aportación")
else:
    gm3.metric("Tiempo estimado", f"{years_needed:.1f} años", f"{months} meses")
gm4.metric("Aportación mensual", f"${monthly_contribution:,.0f}")

if monthly_contribution == 0 and goal_initial < goal_amount:
    st.markdown('<div class="danger-box">🚨 Sin aportación mensual, nunca alcanzarás la meta solo con rendimientos (a menos que el monto inicial ya sea suficiente).</div>', unsafe_allow_html=True)
elif months < max_months:
    fig_goal = go.Figure()
    fig_goal.add_trace(go.Scatter(
        x=list(range(len(history))), y=history, mode="lines",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy", fillcolor=hex_to_rgba(COLORS["primary"], 0.08),
        name="Ahorro acumulado",
        hovertemplate="Mes %{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig_goal.add_hline(y=goal_amount, line_dash="dash", line_color=COLORS["danger"],
                       annotation_text=f"Meta: ${goal_amount:,.0f}")
    fig_goal.update_layout(
        title=f"Progreso hacia «{goal_name}»",
        xaxis_title="Meses", yaxis_title="Ahorro acumulado ($)",
        **base_layout(),
    )
    st.plotly_chart(fig_goal, use_container_width=True)

    st.markdown('<div class="info-box">💡 <strong>Simulador what-if:</strong> incrementa la aportación mensual en la sección anterior para ver cómo se reduce el tiempo necesario.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Proyección patrimonial a largo plazo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📈 Proyección Patrimonial a Largo Plazo</div>', unsafe_allow_html=True)
st.markdown("Compara distintos escenarios de ahorro y rendimiento para tu patrimonio futuro.")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    proj_initial = st.number_input("Patrimonio inicial (MXN)", 0, 10_000_000, 50_000, step=5_000)
with pc2:
    proj_monthly = st.number_input("Aportación mensual (MXN)", 0, 100_000,
                                   max(int(remaining_income), 1_000), step=500, key="proj_monthly")
with pc3:
    proj_years = st.slider("Horizonte (años)", 1, 40, 20)
with pc4:
    st.markdown("**Escenarios de rendimiento:**")
    st.caption("Conservador 5% · Moderado 9% · Agresivo 13%")

scenarios = {
    "Conservador (5%)": 0.05,
    "Moderado (9%)":    0.09,
    "Agresivo (13%)":   0.13,
}
SCEN_COLORS = {"Conservador (5%)": COLORS["neutral"],
               "Moderado (9%)": COLORS["primary"],
               "Agresivo (13%)": COLORS["success"]}

fig_proj = go.Figure()
final_values = {}
for name, rate in scenarios.items():
    proj_df = project_wealth(proj_monthly, rate, proj_years, proj_initial)
    final_values[name] = proj_df["Patrimonio"].iloc[-1]
    fig_proj.add_trace(go.Scatter(
        x=proj_df["Año"], y=proj_df["Patrimonio"], mode="lines",
        name=name, line=dict(color=SCEN_COLORS[name], width=2.5),
        hovertemplate=f"{name}<br>Año %{{x}}<br>$%{{y:,.0f}}<extra></extra>",
    ))

# Línea de aportaciones acumuladas (referencia, igual en todos los escenarios)
ref_df = project_wealth(proj_monthly, 0.0, proj_years, proj_initial)
fig_proj.add_trace(go.Scatter(
    x=ref_df["Año"], y=ref_df["Aportaciones acum."], mode="lines",
    name="Solo aportaciones (sin rendimiento)",
    line=dict(color="#94a3b8", width=1.5, dash="dot"),
))

fig_proj.update_layout(
    title=f"Proyección Patrimonial — {proj_years} años",
    xaxis_title="Año", yaxis_title="Patrimonio ($)",
    legend=dict(orientation="h", y=-0.18),
    **base_layout(),
)
st.plotly_chart(fig_proj, use_container_width=True)

fc1, fc2, fc3 = st.columns(3)
fc1.metric("Conservador (5%)", f"${final_values['Conservador (5%)']:,.0f}")
fc2.metric("Moderado (9%)",    f"${final_values['Moderado (9%)']:,.0f}",
          f"+{(final_values['Moderado (9%)']/final_values['Conservador (5%)']-1):.0%} vs conservador")
fc3.metric("Agresivo (13%)",   f"${final_values['Agresivo (13%)']:,.0f}",
          f"+{(final_values['Agresivo (13%)']/final_values['Conservador (5%)']-1):.0%} vs conservador")

st.markdown(
    '<div class="info-box">📐 <strong>El poder del interés compuesto:</strong> la diferencia entre '
    'escenarios crece exponencialmente con el tiempo, no linealmente. Un punto porcentual adicional '
    'de rendimiento sostenido durante décadas genera una diferencia patrimonial mucho mayor de lo '
    'que la intuición sugiere.</div>',
    unsafe_allow_html=True,
)

with st.expander("📋 Ver tabla de proyección año por año"):
    selected_scenario = st.selectbox("Escenario a detallar", list(scenarios.keys()), index=1)
    detail_df = project_wealth(proj_monthly, scenarios[selected_scenario], proj_years, proj_initial)
    st.dataframe(
        detail_df.style.format({"Patrimonio": "${:,.2f}", "Aportaciones acum.": "${:,.2f}"}),
        use_container_width=True, height=350,
    )
