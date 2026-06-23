import streamlit as st

st.set_page_config(
    page_title="BankInsight Suite",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── Registro de páginas ───────────────────────────────────────────────────────
pg_home     = st.Page("home.py",                           title="Inicio",                  icon="🏦", default=True)
pg_churn    = st.Page("pages/1_Churn.py",                  title="Predicción de Churn",     icon="📉", url_path="churn")
pg_segment  = st.Page("pages/2_Segmentacion.py",           title="Segmentación Clientes",   icon="🎯", url_path="segmentos")
pg_yield    = st.Page("pages/3_Curva_Rendimiento.py",      title="Curva de Rendimiento",    icon="📈", url_path="curva")
pg_restate  = st.Page("pages/4_Precios_Inmuebles.py",      title="Precios de Inmuebles",    icon="🏠", url_path="inmuebles")
pg_sent     = st.Page("pages/5_Sentimiento.py",            title="Sentimiento Financiero",  icon="📰", url_path="sentimiento")
pg_chat     = st.Page("pages/6_Chatbot.py",                title="Chatbot Financiero",      icon="💬", url_path="chatbot")
pg_plan     = st.Page("pages/7_Planeacion.py",             title="Planeación Financiera",   icon="📋", url_path="planeacion")
pg_wealth   = st.Page("pages/8_Wealth.py",                 title="Wealth Management",       icon="💼", url_path="wealth")
pg_earn     = st.Page("pages/9_Ganancias.py",              title="Predicción Ganancias",    icon="📊", url_path="ganancias")
pg_tax      = st.Page("pages/10_Fiscal.py",                title="Optimización Fiscal",     icon="🧾", url_path="fiscal")
pg_docs     = st.Page("pages/11_Documentos.py",            title="Análisis de Documentos",  icon="📄", url_path="documentos")

pg = st.navigation(
    {
        "🏦 BankInsight Suite":    [pg_home],
        "Gestión de Clientes":     [pg_churn, pg_segment, pg_chat],
        "Inversiones y Mercados":  [pg_yield, pg_restate, pg_wealth, pg_earn],
        "Inteligencia Financiera": [pg_sent, pg_docs],
        "Finanzas Personales":     [pg_plan, pg_tax],
    },
    position="sidebar",
    expanded=True,
)

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] { background: #0c1a27; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] p { color: #94a3b8 !important; }

/* Métricas */
[data-testid="stMetric"] {
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-radius: 12px; padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { font-size:0.78rem; color:#0369a1; font-weight:600; }
[data-testid="stMetricValue"] { font-size:1.55rem; font-weight:700; color:#0c4a6e; }
[data-testid="stMetricDelta"] { font-size:0.75rem; }

/* Clases de utilidad */
.section-title {
    font-size:1.05rem; font-weight:700; color:#0f172a;
    border-left:4px solid #0ea5e9; padding-left:12px;
    margin:1.5rem 0 1rem;
}
.info-box    { background:#f0f9ff; border:1px solid #7dd3fc; border-radius:10px; padding:11px 15px; font-size:13px; color:#0c4a6e; margin:8px 0; }
.warn-box    { background:#fffbeb; border:1px solid #fcd34d; border-radius:10px; padding:11px 15px; font-size:13px; color:#78350f; margin:8px 0; }
.danger-box  { background:#fef2f2; border:1px solid #fca5a5; border-radius:10px; padding:11px 15px; font-size:13px; color:#7f1d1d; margin:8px 0; }
.success-box { background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:11px 15px; font-size:13px; color:#14532d; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

pg.run()
