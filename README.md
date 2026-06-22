# 🏦 BankInsight Suite — Banca y Gestión de Clientes

Plataforma de analítica para banca y gestión de clientes con 11 módulos de ML, NLP
y simulación financiera. Stack 100% open source.

---

## 📦 Módulos

| # | Módulo | Modelo / Técnica | Categoría |
|---|--------|-------------------|-----------|
| 1 | Predicción de Churn | LightGBM + SMOTE + SHAP | Clientes |
| 2 | Segmentación de Clientes | K-Means + Elbow Method | Clientes |
| 3 | Curva de Rendimiento | Análisis de escenarios + pricing de bonos | Inversiones |
| 4 | Precios de Inmuebles | XGBoost Regressor + SHAP | Inversiones |
| 5 | Sentimiento Financiero | Léxico financiero propio (español, 88% precisión) | Inteligencia |
| 6 | Chatbot Financiero | Motor de reglas + NLP + simuladores | Clientes |
| 7 | Planeación Financiera | Presupuesto + metas + interés compuesto | Finanzas personales |
| 8 | Wealth Management | KNN + asset allocation | Inversiones |
| 9 | Predicción de Ganancias | XGBoost Regressor + SHAP | Inversiones |
| 10 | Optimización Fiscal | Simulador ISR 2024 (LISR Art. 96/151) | Finanzas personales |
| 11 | Análisis de Documentos | NLP + regex (extracción de KPIs) | Inteligencia |

---

## 🚀 Ejecutar localmente

```bash
cd bank_app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Estructura

```
bank_app/
├── app.py              ← st.navigation() — entry point
├── home.py             ← Landing page
├── requirements.txt
├── .streamlit/config.toml
├── utils/
│   ├── __init__.py
│   └── helpers.py       ← Generadores de datos + modelos + plots (capa compartida)
└── pages/
    ├── 1_Churn.py
    ├── 2_Segmentacion.py
    ├── 3_Curva_Rendimiento.py
    ├── 4_Precios_Inmuebles.py
    ├── 5_Sentimiento.py
    ├── 6_Chatbot.py
    ├── 7_Planeacion.py
    ├── 8_Wealth.py
    ├── 9_Ganancias.py
    ├── 10_Fiscal.py
    └── 11_Documentos.py
```

---

## ⚠️ Notas importantes

- **Datos sintéticos**: todos los datasets se generan con semillas fijas (reproducibles). No son datos reales de ningún banco o institución.
- **Optimización Fiscal**: usa la tarifa LISR 2024 vigente para asalariados; es educativo y no sustituye asesoría fiscal profesional.
- **Sentimiento Financiero**: usa un léxico financiero en español construido a medida (VADER y TextBlob no rinden bien en español financiero — se validó empíricamente).
- **Nombres de archivo sin emojis**: por compatibilidad con Git en Windows/Mac y con el sistema de routing de Streamlit Cloud.

---

## 🔧 Stack tecnológico

| Componente | Librería |
|-----------|----------|
| Frontend | Streamlit (`st.navigation` + `st.Page`) |
| Modelos ML | XGBoost, LightGBM, scikit-learn |
| Balanceo de clases | imbalanced-learn (SMOTE) |
| Explicabilidad | SHAP |
| Visualizaciones | Plotly |
| NLP | Regex + léxico propio (sin dependencias pesadas) |
