import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZZATO (STILE PROFESSIONAL) ---
st.markdown("""
    <style>
    /* Sfondo generale */
    .stApp {
        background-color: #f4f6f9;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    /* Titoli */
    h1, h2, h3 {
        color: #0f2942; /* Blu scuro istituzionale */
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Bottone Principale */
    div.stButton > button {
        background-color: #0f2942;
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 4px;
        width: 100%;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #1c4b75;
        color: white;
    }
    /* Box Risultati */
    .result-box-success {
        padding: 20px;
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 5px;
        color: #1b5e20;
    }
    .result-box-fail {
        padding: 20px;
        background-color: #ffebee;
        border-left: 5px solid #c62828;
        border-radius: 5px;
        color: #b71c1c;
    }
    </style>
    """, unsafe_allow_html=True)


# --- CARICAMENTO ASSET ---
@st.cache_resource
def load_assets():
    model_app = joblib.load('modello_mutui.pkl')
    model_bank = joblib.load('modello_banche.pkl')
    scaler = joblib.load('scaler_mutui.pkl')
    columns = joblib.load('colonne_mutui.pkl')
    encoder = joblib.load('encoder_banche.pkl')
    return model_app, model_bank, scaler, columns, encoder


try:
    model_app, model_bank, scaler, model_columns, bank_encoder = load_assets()
except:
    st.error("Errore Critico: File del modello non trovati. Eseguire lo script di training.")
    st.stop()

# --- HEADER ---
st.title("Credit Scoring & Matching System")
st.markdown("---")

# --- SIDEBAR (INPUT FORMALI) ---
with st.sidebar:
    st.markdown("### 📋 Parametri Pratica")

    st.markdown("**Dati Economici**")
    reddito = st.number_input("Reddito Lordo Annuo (€)", value=45000, step=1000)
    importo = st.number_input("Importo Richiesto (€)", value=120000, step=5000)
    durata = st.slider("Ammortamento (Anni)", 5, 30, 20)

    st.markdown("---")
    st.markdown("**Profilo di Rischio**")
    score_crif = st.slider("Credit Score (CRIF)", 300, 900, 650)
    crif_negativo = st.radio("Segnalazioni in Centrale Rischi", ["Assenti", "Presenti"])

    st.markdown("---")
    st.markdown("**Situazione Patrimoniale**")
    patrimonio = st.number_input("Patrimonio Complessivo (€)", value=60000)

    with st.expander("Dettagli Anagrafici"):
        familiari = st.number_input("Familiari a carico", 0, 10, 1)
        istruzione = st.selectbox("Livello Istruzione", ["Laurea", "Diploma o inferiore"])
        lavoro = st.selectbox("Posizione Lavorativa", ["Dipendente", "Autonomo"])

# --- PRE-ELABORAZIONE ---
rata = (importo / (durata * 12)) * 1.05
ratio = (rata * 12) / reddito if reddito > 0 else 0
score_norm = (score_crif - 300) / 600

# Costruzione DataFrame
input_dict = {
    'familiari_a_carico': familiari,
    'reddito_annuo': reddito,
    'importo_mutuo': importo,
    'durata_mutuo_anni': durata,
    'score_crif': score_crif,
    'patrimonio_residenziale': patrimonio * 0.7,
    'patrimonio_commerciale': 0,
    'beni_lusso': 0,
    'liquidita_bancaria': patrimonio * 0.3,
    'rata_mensile': rata,
    'rapporto_rata_reddito': ratio,
    'score_crif_norm': score_norm,
    'crif_negativo': 1 if crif_negativo == "Presenti" else 0,
    'livello_istruzione_Diploma_o_inferiore': 1 if istruzione != "Laurea" else 0,
    'livello_istruzione_Laurea': 1 if istruzione == "Laurea" else 0,
    'tipo_lavoro_Autonomo': 1 if lavoro == "Autonomo" else 0,
    'tipo_lavoro_Dipendente': 1 if lavoro == "Dipendente" else 0
}

df_in = pd.DataFrame([input_dict])
df_in = df_in.reindex(columns=model_columns, fill_value=0)
X_scaled = scaler.transform(df_in)

# --- DASHBOARD PRINCIPALE ---
col1, col2, col3 = st.columns(3)
col1.metric("Rapporto Rata/Reddito", f"{ratio:.1%}", delta_color="inverse")
col2.metric("Rating Cliente", f"{score_crif}/900")
col3.metric("Rata Mensile Stimata", f"€ {rata:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# --- ESECUZIONE ---
if st.button("CALCOLA MERITO CREDITIZIO"):

    # Spinner professionale invece dei palloncini
    with st.spinner('Elaborazione algoritmi di rischio in corso...'):
        pred_app = model_app.predict(X_scaled)[0]
        prob_app = model_app.predict_proba(X_scaled)[0][1]

    st.markdown("### Risultato Valutazione")

    if pred_app == 1:
        # Recupero nome banca
        pred_bank_idx = model_bank.predict(X_scaled)[0]
        bank_name = bank_encoder.inverse_transform([pred_bank_idx])[0]
        bank_clean = bank_name.replace('_', ' ')

        # HTML personalizzato per box verde elegante
        st.markdown(f"""
        <div class="result-box-success">
            <h3>✅ ESITO: POSITIVO</h3>
            <p>La pratica presenta parametri compatibili con l'erogazione.</p>
            <p><strong>Probabilità di Approvazione:</strong> {prob_app:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Matching Istituti Bancari")
        st.info(f"In base al profilo storico, l'istituto con maggiore affinità è: **{bank_clean}**")

    else:
        # HTML personalizzato per box rosso elegante
        st.markdown(f"""
        <div class="result-box-fail">
            <h3>⚠️ ESITO: NEGATIVO</h3>
            <p>La pratica presenta indicatori di rischio elevati.</p>
            <p><strong>Probabilità di Approvazione:</strong> {prob_app:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Note di Analisi")
        if ratio > 0.35:
            st.warning("Il rapporto Rata/Reddito eccede i limiti prudenziali (>35%).")
        if score_crif < 600:
            st.warning("Lo storico creditizio (CRIF) risulta sotto la soglia di sicurezza.")
        if crif_negativo == "Presenti":
            st.error("Rilevate segnalazioni negative in centrale rischi.")