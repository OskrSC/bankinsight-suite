"""
Módulo 6 — Chatbot Financiero
Interfaz conversacional con st.chat_message · base de conocimiento por categoría ·
detección de intención + extracción de montos · simuladores de inversión y crédito
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.helpers import (
    CHATBOT_KB, chatbot_respond,
    simulate_investment, simulate_loan_payment,
    hex_to_rgba, base_layout, COLORS,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f0f9ff;border:1px solid #bae6fd;
  border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.05rem;font-weight:700;color:#0f172a;
  border-left:4px solid #0ea5e9;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f0f9ff;border:1px solid #7dd3fc;border-radius:10px;
  padding:11px 15px;font-size:13px;color:#0c4a6e;margin:8px 0}
.cat-pill{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;
  border-radius:20px;margin:2px}
</style>
""", unsafe_allow_html=True)

CAT_COLORS = {
    "cuenta":      COLORS["primary"],
    "crédito":     COLORS["warning"],
    "inversión":   COLORS["success"],
    "operaciones": COLORS["purple"],
    "tarjeta":     "#0891b2",
    "seguros":     COLORS["danger"],
    "sucursal":    COLORS["neutral"],
    "general":     COLORS["neutral"],
}

QUICK_PROMPTS = [
    "¿Cuál es mi saldo?",
    "¿Cuánto crédito me pueden prestar?",
    "Quiero invertir 50000 pesos",
    "¿Dónde está la sucursal más cercana?",
    "Información de mi tarjeta de crédito",
    "¿Qué seguros tengo activos?",
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 💬 Chatbot Financiero")
st.markdown(
    "Asistente conversacional con **detección de intención + extracción de montos**. "
    "Responde consultas de saldo, crédito, inversiones, tarjetas, seguros y sucursales, "
    "y simula operaciones financieras en tiempo real."
)
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Categorías cubiertas", str(len(CAT_COLORS) - 1))
c2.metric("Simuladores activos",  "2", "Inversión + Crédito")
c3.metric("Detección de montos",  "Sí", "Regex + unidades (k, mil, M)")
c4.metric("Motor",                "Reglas + NLP", "100% local, sin API externa")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Chat interactivo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">💬 Conversación</div>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "¡Hola! Soy tu asistente financiero. Puedo ayudarte con saldo, "
                    "crédito, inversiones, tarjetas, seguros y sucursales. ¿En qué te ayudo?",
         "category": "general"}
    ]

# Prompts rápidos
st.markdown("**Prueba con una de estas preguntas:**")
qp_cols = st.columns(3)
for i, prompt in enumerate(QUICK_PROMPTS):
    col = qp_cols[i % 3]
    if col.button(prompt, key=f"qp_{i}", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        result = chatbot_respond(prompt)
        st.session_state.chat_history.append({
            "role": "assistant", "content": result["response"],
            "category": result["category"],
        })
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Historial de chat
chat_container = st.container(height=420, border=True)
with chat_container:
    for msg in st.session_state.chat_history:
        avatar = "🧑" if msg["role"] == "user" else "🏦"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("category"):
                color = CAT_COLORS.get(msg["category"], COLORS["neutral"])
                st.markdown(
                    f'<span class="cat-pill" style="background:{hex_to_rgba(color,.12)};'
                    f'color:{color}">{msg["category"].upper()}</span>',
                    unsafe_allow_html=True,
                )

# Input de usuario
user_msg = st.chat_input("Escribe tu consulta financiera (ej. 'quiero invertir 30000 a 90 días')")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    result = chatbot_respond(user_msg)
    st.session_state.chat_history.append({
        "role": "assistant", "content": result["response"],
        "category": result["category"],
    })
    st.rerun()

col_clear, _ = st.columns([1, 4])
if col_clear.button("🗑️ Limpiar conversación"):
    st.session_state.chat_history = [st.session_state.chat_history[0]]
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Simuladores financieros
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">🧮 Simuladores Financieros</div>', unsafe_allow_html=True)
st.markdown("Estos son los mismos cálculos que el chatbot ejecuta automáticamente cuando detecta un monto en tu mensaje.")

tab_inv, tab_loan = st.tabs(["📈 Simulador de Inversión", "💳 Simulador de Crédito"])

with tab_inv:
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        inv_amount = st.number_input("Monto a invertir (MXN)", 100, 5_000_000, 50_000, step=1_000)
    with ic2:
        inv_rate = st.slider("Tasa anual (%)", 5.0, 15.0, 11.5, step=0.1) / 100
    with ic3:
        inv_days = st.selectbox("Plazo (días)", [28, 90, 180, 360], index=1)

    sim_inv = simulate_investment(inv_amount, inv_rate, inv_days)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Interés bruto",       f"${sim_inv['interest_gross']:,.2f}")
    r2.metric("ISR retenido (20%)",  f"-${sim_inv['isr']:,.2f}")
    r3.metric("Interés neto",        f"${sim_inv['interest_net']:,.2f}")
    r4.metric("Saldo final",         f"${sim_inv['final_balance']:,.2f}",
              f"+{sim_inv['interest_net']/inv_amount:.2%}")

    # Gráfica de crecimiento día a día
    days_range = list(range(0, inv_days + 1, max(1, inv_days // 30)))
    balances = [simulate_investment(inv_amount, inv_rate, d)["final_balance"] for d in days_range]
    fig_inv = go.Figure(go.Scatter(
        x=days_range, y=balances, mode="lines+markers",
        line=dict(color=COLORS["success"], width=2.5),
        fill="tozeroy", fillcolor=hex_to_rgba(COLORS["success"], 0.08),
        hovertemplate="Día %{x}<br>Saldo: $%{y:,.2f}<extra></extra>",
    ))
    fig_inv.update_layout(
        title=f"Crecimiento de la inversión — ${inv_amount:,.0f} a {inv_rate*100:.1f}% anual",
        xaxis_title="Días", yaxis_title="Saldo neto ($)",
        **base_layout(),
    )
    st.plotly_chart(fig_inv, use_container_width=True)

with tab_loan:
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        loan_amount = st.number_input("Monto del crédito (MXN)", 1_000, 1_000_000, 80_000, step=1_000)
    with lc2:
        loan_rate = st.slider("Tasa anual (%)", 8.0, 35.0, 18.5, step=0.5) / 100
    with lc3:
        loan_months = st.selectbox("Plazo (meses)", [12, 24, 36, 48, 60], index=2)

    sim_loan = simulate_loan_payment(loan_amount, loan_rate, loan_months)

    r1, r2, r3 = st.columns(3)
    r1.metric("Pago mensual",      f"${sim_loan['monthly_payment']:,.2f}")
    r2.metric("Total a pagar",     f"${sim_loan['total_paid']:,.2f}")
    r3.metric("Intereses totales", f"${sim_loan['total_interest']:,.2f}",
              f"{sim_loan['total_interest']/loan_amount:.1%} del monto")

    # Tabla de amortización simplificada (primeros y últimos periodos)
    r_m = loan_rate / 12
    balance = loan_amount
    rows = []
    for m in range(1, loan_months + 1):
        interest_pmt = balance * r_m
        principal_pmt = sim_loan["monthly_payment"] - interest_pmt
        balance -= principal_pmt
        rows.append({"Mes": m, "Pago": sim_loan["monthly_payment"],
                     "Interés": interest_pmt, "Capital": principal_pmt,
                     "Saldo restante": max(balance, 0)})
    amort_df = pd.DataFrame(rows)

    fig_amort = go.Figure()
    fig_amort.add_trace(go.Bar(
        x=amort_df["Mes"], y=amort_df["Interés"], name="Interés",
        marker_color=COLORS["danger"],
    ))
    fig_amort.add_trace(go.Bar(
        x=amort_df["Mes"], y=amort_df["Capital"], name="Capital",
        marker_color=COLORS["primary"],
    ))
    fig_amort.update_layout(
        barmode="stack",
        title="Composición del Pago Mensual — Interés vs Capital",
        xaxis_title="Mes", yaxis_title="Monto ($)",
        legend=dict(orientation="h", y=-0.18),
        **base_layout(),
    )
    st.plotly_chart(fig_amort, use_container_width=True)

    with st.expander("📋 Ver tabla de amortización completa"):
        st.dataframe(
            amort_df.style.format({
                "Pago": "${:,.2f}", "Interés": "${:,.2f}",
                "Capital": "${:,.2f}", "Saldo restante": "${:,.2f}",
            }),
            use_container_width=True, height=350,
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Explorador de la base de conocimiento
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📚 Base de Conocimiento del Chatbot</div>', unsafe_allow_html=True)
st.markdown("Categorías y palabras clave que el motor de reglas reconoce para enrutar cada consulta.")

kb_rows = []
for key, val in CHATBOT_KB.items():
    if key == "default":
        continue
    kb_rows.append({
        "Categoría": val["category"],
        "Palabras clave": ", ".join(val["patterns"]),
        "Ejemplo de respuesta": val["response"][:90].replace("\n", " ") + "…",
    })
kb_df = pd.DataFrame(kb_rows)

ca, cb = st.columns([1, 2])
with ca:
    cat_counts = pd.Series([r["Categoría"] for r in kb_rows]).value_counts()
    fig_kb = go.Figure(go.Bar(
        x=[len(v["patterns"]) for v in CHATBOT_KB.values() if v != CHATBOT_KB["default"]],
        y=[v["category"] for k, v in CHATBOT_KB.items() if k != "default"],
        orientation="h",
        marker_color=[CAT_COLORS.get(v["category"], COLORS["neutral"])
                     for k, v in CHATBOT_KB.items() if k != "default"],
        hovertemplate="%{y}: %{x} palabras clave<extra></extra>",
    ))
    fig_kb.update_layout(
        title="Palabras clave por categoría",
        xaxis_title="Número de patrones",
        height=320, paper_bgcolor="white", plot_bgcolor=COLORS["bg"],
        margin=dict(t=44, b=44, l=110, r=20),
    )
    st.plotly_chart(fig_kb, use_container_width=True)
with cb:
    st.dataframe(kb_df, use_container_width=True, height=320, hide_index=True)

st.markdown(
    '<div class="info-box">💡 <strong>Cómo funciona:</strong> el motor primero busca '
    'intención de simulación (monto + palabra como "invertir" o "crédito"); si la '
    'detecta, ejecuta el cálculo financiero real. Si no, busca coincidencias de '
    'palabras clave en las categorías de la tabla. En producción, esto se sustituiría '
    'por un modelo de NLU (Dialogflow, Rasa) conectado al core bancario real.</div>',
    unsafe_allow_html=True,
)
