import streamlit as st

st.markdown("# 🏦 BankInsight Suite")
st.markdown(
    "**Plataforma de Analítica para Banca y Gestión de Clientes** — "
    "11 módulos de ML, NLP y simulación financiera sobre datos sintéticos reproducibles."
)
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Módulos",         "11",         "Cobertura integral")
c2.metric("Modelos ML/NLP",  "8",          "Explicables con SHAP")
c3.metric("Variables",       "45+",        "Multi-dimensionales")
c4.metric("Stack",           "Open Source","100% gratuito")

st.markdown("---")
st.markdown('<div class="section-title">Módulos disponibles</div>',
            unsafe_allow_html=True)

modules = [
    # (categoria, icono, nombre, descripción)
    ("Clientes", "📉", "Predicción de Churn",
     "LightGBM + SMOTE + SHAP. Identifica clientes con riesgo de abandono y genera plan de retención."),
    ("Clientes", "🎯", "Segmentación de Clientes",
     "K-Means clustering. Agrupa en Premium / Activo / Ocasional / Inactivo con acciones de marketing."),
    ("Clientes", "💬", "Chatbot Financiero",
     "Motor de reglas + NLP. Consultas de saldo, crédito, inversiones y sucursales con simuladores integrados."),
    ("Inversiones", "📈", "Curva de Rendimiento",
     "Análisis interactivo por escenario (normal, invertida, crisis). Calculadora de precio de bono."),
    ("Inversiones", "🏠", "Precios de Inmuebles",
     "XGBoost Regressor + SHAP. Valuación por zona, m², amenidades y distancia al metro."),
    ("Inversiones", "💼", "Wealth Management",
     "KNN personalizado. Recomienda estrategia conservadora / balanceada / agresiva por perfil."),
    ("Inversiones", "📊", "Predicción de Ganancias",
     "XGBoost Regressor. Proyecta EPS corporativo con revenue, márgenes y sentimiento de mercado."),
    ("Inteligencia", "📰", "Sentimiento Financiero",
     "Léxico financiero propio en español (88% precisión validada). Clasifica noticias y detecta tendencias."),
    ("Inteligencia", "📄", "Análisis de Documentos",
     "NLP + regex. Extrae KPIs, entidades financieras y genera resumen ejecutivo de reportes anuales."),
    ("Finanzas personales", "📋", "Planeación Financiera",
     "Presupuesto mensual, metas de ahorro y proyección patrimonial por escenario."),
    ("Finanzas personales", "🧾", "Optimización Fiscal",
     "Simulador ISR 2024 (LISR Art. 96 y 151). Calcula ahorro fiscal aplicando topes legales reales."),
]

cat_colors = {
    "Clientes": "#0ea5e9", "Inversiones": "#16a34a",
    "Inteligencia": "#7c3aed", "Finanzas personales": "#d97706",
}

for i in range(0, len(modules), 3):
    cols = st.columns(3)
    for col, (cat, icon, name, desc) in zip(cols, modules[i:i+3]):
        color = cat_colors[cat]
        with col:
            st.markdown(f"""
            <div style='border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;height:100%'>
              <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>
                <span style='font-size:1.1rem'>{icon}</span>
                <span style='font-size:13px;font-weight:700;color:#0f172a'>{name}</span>
              </div>
              <span style='font-size:10px;background:{color}18;color:{color};
                    border:1px solid {color}44;border-radius:20px;padding:1px 8px;font-weight:600'>
                {cat}
              </span>
              <p style='font-size:12px;color:#475569;margin:8px 0 0'>{desc}</p>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    '<div class="info-box">💡 <strong>Datos sintéticos</strong> generados con semillas '
    'fijas (reproducibles). En producción: conectar con CRM, core bancario o APIs de mercado. '
    'Todas las cifras son ilustrativas.</div>',
    unsafe_allow_html=True,
)
