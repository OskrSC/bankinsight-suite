"""
Módulo 10 — Optimización Fiscal
Simulador ISR 2024 (LISR Art. 96) · deducciones personales con tope legal (Art. 151) ·
comparativa antes/después · tabla de tarifa progresiva
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.helpers import (
    ISR_BRACKETS_2024, calc_isr_anual, calc_deducciones,
    DEDUCTION_CATEGORIES, COLEGIATURA_LIMITS_2024, UMA_ANUAL_2024,
    hex_to_rgba, base_layout, COLORS,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #0891b2;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
.warn-box{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#78350f;margin:8px 0}
.success-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#14532d;margin:8px 0}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧾 Optimización Fiscal")
st.markdown(
    "Simulador de **ISR 2024** (LISR Artículo 96, tabla de asalariados) con cálculo "
    "de deducciones personales aplicando el **tope legal real** (Artículo 151: el "
    "menor entre 15% del ingreso anual y 5 UMA)."
)
st.markdown(
    f'<div class="warn-box">⚠️ Este simulador es educativo y usa la tarifa LISR 2024 '
    f'vigente para asalariados. UMA diaria 2024 = $108.57 (UMA anual = '
    f'${UMA_ANUAL_2024:,.2f}). No sustituye asesoría fiscal profesional ni considera '
    f'subsidio al empleo, regímenes especiales o reformas posteriores.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Calculadora de ISR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">💰 Calculadora de ISR Anual</div>', unsafe_allow_html=True)

ingreso_anual = st.number_input(
    "Ingreso anual gravable (MXN)", 10_000, 10_000_000, 480_000, step=10_000,
    help="Suma de sueldos, salarios y prestaciones gravadas del año fiscal",
)

isr_sin_deducciones = calc_isr_anual(ingreso_anual)
tasa_efectiva_sin = isr_sin_deducciones / ingreso_anual if ingreso_anual > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Ingreso anual",        f"${ingreso_anual:,.0f}")
c2.metric("ISR causado (sin deducciones)", f"${isr_sin_deducciones:,.2f}")
c3.metric("Tasa efectiva",        f"{tasa_efectiva_sin:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Deducciones personales
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📋 Deducciones Personales (Art. 151 LISR)</div>',
            unsafe_allow_html=True)
st.markdown("Ingresa tus gastos deducibles del año. El sistema aplicará automáticamente el tope legal.")

with st.expander("ℹ️ Ver topes y reglas por categoría"):
    rules_df = pd.DataFrame([
        {"Categoría": k, "Regla específica": v} for k, v in DEDUCTION_CATEGORIES.items()
    ])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)
    st.markdown("**Límites de colegiaturas por nivel (anual, por hijo):**")
    col_df = pd.DataFrame([
        {"Nivel": k, "Límite anual": v} for k, v in COLEGIATURA_LIMITS_2024.items()
    ])
    st.dataframe(
        col_df.style.format({"Límite anual": "${:,.0f}"}),
        use_container_width=True, hide_index=True,
    )

ded_cols = st.columns(2)
gastos = {}
categories = list(DEDUCTION_CATEGORIES.keys())
for i, cat in enumerate(categories):
    col = ded_cols[i % 2]
    gastos[cat] = col.number_input(cat, 0, 500_000, 0, step=500, key=f"ded_{i}")

deduccion_info = calc_deducciones(gastos, ingreso_anual)

dc1, dc2, dc3, dc4 = st.columns(4)
dc1.metric("Suma de gastos",          f"${deduccion_info['suma_bruta']:,.0f}")
dc2.metric("Tope legal aplicable",    f"${deduccion_info['tope_aplicable']:,.0f}",
          "Menor entre 15% ingreso y 5 UMA")
dc3.metric("Deducción aplicada",      f"${deduccion_info['deduccion_aplicada']:,.0f}")
dc4.metric("Excedente no deducible",  f"${deduccion_info['excedente_no_deducible']:,.0f}",
          delta_color="inverse" if deduccion_info["excedente_no_deducible"] > 0 else "off")

if deduccion_info["excedente_no_deducible"] > 0:
    st.markdown(
        f'<div class="warn-box">⚠️ Tus deducciones (${deduccion_info["suma_bruta"]:,.0f}) '
        f'superan el tope legal aplicable. Solo puedes deducir '
        f'${deduccion_info["deduccion_aplicada"]:,.0f}; el resto '
        f'(${deduccion_info["excedente_no_deducible"]:,.0f}) no es deducible este año.</div>',
        unsafe_allow_html=True,
    )

# Barra visual del tope
fig_cap = go.Figure()
fig_cap.add_trace(go.Bar(
    y=["Deducciones"], x=[deduccion_info["deduccion_aplicada"]],
    orientation="h", name="Aplicada", marker_color=COLORS["success"],
))
fig_cap.add_trace(go.Bar(
    y=["Deducciones"], x=[deduccion_info["excedente_no_deducible"]],
    orientation="h", name="Excedente (no deducible)", marker_color=COLORS["danger"],
    base=[deduccion_info["deduccion_aplicada"]],
))
fig_cap.add_vline(x=deduccion_info["tope_aplicable"], line_dash="dash",
                  line_color="#0f172a", annotation_text="Tope legal")
fig_cap.update_layout(
    barmode="stack", title="Deducción Aplicada vs Tope Legal",
    xaxis_title="Monto ($)", height=180,
    paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
    margin=dict(t=40, b=30, l=80, r=20),
    legend=dict(orientation="h", y=-0.3),
)
st.plotly_chart(fig_cap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Comparativa antes/después
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">⚖️ Impacto de las Deducciones en tu ISR</div>',
            unsafe_allow_html=True)

base_gravable_despues = max(ingreso_anual - deduccion_info["deduccion_aplicada"], 0)
isr_con_deducciones   = calc_isr_anual(base_gravable_despues)
ahorro_fiscal         = isr_sin_deducciones - isr_con_deducciones
tasa_efectiva_con     = isr_con_deducciones / ingreso_anual if ingreso_anual > 0 else 0

r1, r2, r3, r4 = st.columns(4)
r1.metric("ISR sin deducciones",  f"${isr_sin_deducciones:,.2f}")
r2.metric("ISR con deducciones",  f"${isr_con_deducciones:,.2f}",
          f"-${ahorro_fiscal:,.2f}", delta_color="inverse")
r3.metric("Ahorro fiscal",        f"${ahorro_fiscal:,.2f}",
          f"{ahorro_fiscal/isr_sin_deducciones:.1%} del ISR original" if isr_sin_deducciones > 0 else "")
r4.metric("Nueva tasa efectiva",  f"{tasa_efectiva_con:.2%}",
          f"{tasa_efectiva_con - tasa_efectiva_sin:+.2%}")

if ahorro_fiscal > 0:
    st.markdown(
        f'<div class="success-box">✅ Tus deducciones generan un ahorro fiscal de '
        f'<strong>${ahorro_fiscal:,.2f}</strong>, equivalente a '
        f'{ahorro_fiscal/isr_sin_deducciones:.1%} de tu ISR original.</div>',
        unsafe_allow_html=True,
    )

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    x=["Sin deducciones", "Con deducciones"],
    y=[isr_sin_deducciones, isr_con_deducciones],
    marker_color=[COLORS["danger"], COLORS["success"]],
    text=[f"${v:,.0f}" for v in [isr_sin_deducciones, isr_con_deducciones]],
    textposition="outside",
))
fig_comp.update_layout(
    title="ISR Anual: Antes vs Después de Deducciones",
    yaxis_title="ISR ($)",
    **base_layout(),
)
st.plotly_chart(fig_comp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Tabla de tarifa progresiva y curva marginal
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📊 Tarifa Progresiva LISR 2024</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋 Tabla de tarifas", "📈 Curva de tasa marginal vs efectiva"])

with tab1:
    bracket_df = pd.DataFrame(
        ISR_BRACKETS_2024,
        columns=["Límite inferior", "Límite superior", "Cuota fija", "Tasa marginal %"],
    )
    bracket_df["Límite superior"] = bracket_df["Límite superior"].replace(
        float("inf"), np.nan
    )
    # Marcar el rango que aplica al ingreso actual
    def highlight_bracket(row):
        lo, hi = row["Límite inferior"], row["Límite superior"]
        hi_check = hi if not pd.isna(hi) else float("inf")
        if lo <= ingreso_anual <= hi_check:
            return ["background-color: #dbeafe"] * len(row)
        return [""] * len(row)

    st.dataframe(
        bracket_df.style
        .format({"Límite inferior": "${:,.2f}", "Límite superior": "${:,.2f}",
                 "Cuota fija": "${:,.2f}", "Tasa marginal %": "{:.2f}%"}, na_rep="Sin límite")
        .apply(highlight_bracket, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.markdown('<div class="info-box">La fila resaltada en azul corresponde al rango donde cae tu ingreso anual actual. La <strong>tasa marginal</strong> solo aplica al excedente sobre el límite inferior, no a todo el ingreso.</div>', unsafe_allow_html=True)

with tab2:
    income_range = np.linspace(10_000, 2_000_000, 200)
    isr_values   = [calc_isr_anual(inc) for inc in income_range]
    effective_rates = [isr/inc*100 for isr, inc in zip(isr_values, income_range)]
    marginal_rates  = []
    for inc in income_range:
        for lo, hi, cuota, tasa in ISR_BRACKETS_2024:
            if lo <= inc <= hi:
                marginal_rates.append(tasa)
                break
        else:
            marginal_rates.append(ISR_BRACKETS_2024[-1][3])

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=income_range, y=effective_rates, mode="lines",
        name="Tasa efectiva", line=dict(color=COLORS["primary"], width=2.5),
    ))
    fig_curve.add_trace(go.Scatter(
        x=income_range, y=marginal_rates, mode="lines",
        name="Tasa marginal", line=dict(color=COLORS["danger"], width=1.5, dash="dash"),
    ))
    fig_curve.add_vline(x=ingreso_anual, line_dash="dot", line_color="#0f172a",
                        annotation_text="Tu ingreso")
    fig_curve.update_layout(
        title="Tasa Efectiva vs Tasa Marginal por Nivel de Ingreso",
        xaxis_title="Ingreso anual ($)", yaxis_title="Tasa (%)",
        legend=dict(orientation="h", y=-0.18),
        **base_layout(),
    )
    st.plotly_chart(fig_curve, use_container_width=True)
    st.markdown(
        '<div class="info-box">📐 La <strong>tasa efectiva</strong> (ISR/ingreso total) '
        'siempre es menor que la <strong>tasa marginal</strong> (la del último peso ganado) '
        'en un sistema progresivo — por eso ganar más nunca reduce tu ingreso neto total, '
        'aunque el siguiente peso se grave a una tasa más alta.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Simulador de escenarios de planeación
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🎯 Simulador: ¿Cuánto Debo Deducir para Maximizar mi Ahorro?</div>',
            unsafe_allow_html=True)

ded_range = np.linspace(0, min(5 * UMA_ANUAL_2024, ingreso_anual * 0.30), 50)
savings_curve = []
for d in ded_range:
    base_d = max(ingreso_anual - d, 0)
    isr_d  = calc_isr_anual(base_d)
    savings_curve.append(isr_sin_deducciones - isr_d)

fig_opt = go.Figure(go.Scatter(
    x=ded_range, y=savings_curve, mode="lines",
    line=dict(color=COLORS["success"], width=2.5),
    fill="tozeroy", fillcolor=hex_to_rgba(COLORS["success"], 0.08),
    hovertemplate="Deducción: $%{x:,.0f}<br>Ahorro: $%{y:,.0f}<extra></extra>",
))
tope_actual = deduccion_info["tope_aplicable"]
fig_opt.add_vline(x=tope_actual, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text=f"Tu tope: ${tope_actual:,.0f}")
fig_opt.update_layout(
    title="Ahorro Fiscal Potencial según Monto Deducido",
    xaxis_title="Monto deducido ($)", yaxis_title="Ahorro en ISR ($)",
    **base_layout(),
)
st.plotly_chart(fig_opt, use_container_width=True)

st.markdown(
    f'<div class="info-box">💡 <strong>Recomendación:</strong> dado tu ingreso de '
    f'${ingreso_anual:,.0f}, el tope legal de deducción es '
    f'<strong>${tope_actual:,.0f}</strong>. Maximizar deducciones hasta ese punto '
    f'(con gastos reales y comprobables) genera el mayor ahorro fiscal posible dentro '
    f'del marco legal.</div>',
    unsafe_allow_html=True,
)
