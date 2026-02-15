import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# 1. Carica i dati
try:
    df = pd.read_csv('dati.csv')
    print("✅ File caricato!")
except FileNotFoundError:
    print("❌ ERRORE: File 'dati.csv' non trovato.")
    exit()

# 2. Pulizia
if 'id_mutuo' in df.columns: df = df.drop(columns=['id_mutuo'])

# --- PREPARAZIONE MODELLO 1: APPROVAZIONE (Sì/No) ---
# Rimuoviamo la banca per questo modello perché non deve barare sapendo la banca
df_approval = df.drop(columns=['banca_consigliata'])

# Gestione colonne (come prima)
if 'livello_istruzione_Laurea' in df_approval.columns:
    df_approval['livello_istruzione_Diploma_o_inferiore'] = 1 - df_approval['livello_istruzione_Laurea']
if 'tipo_lavoro_Autonomo' in df_approval.columns:
    df_approval['tipo_lavoro_Dipendente'] = 1 - df_approval['tipo_lavoro_Autonomo']

X_app = df_approval.drop(columns=['mutuo_approvato'])
y_app = df_approval['mutuo_approvato']

scaler = MinMaxScaler()
X_app_scaled = scaler.fit_transform(X_app)

print("⏳ Addestro Modello Approvazione...")
model_approval = RandomForestClassifier(max_depth=7, n_estimators=100, random_state=42)
model_approval.fit(X_app_scaled, y_app)

# --- PREPARAZIONE MODELLO 2: CONSIGLIO BANCA ---
# Usiamo solo i mutui APPROVATI per imparare quale banca accetta chi
df_bank = df[df['mutuo_approvato'] == 1].copy()

# Se 'banca_consigliata' è "Nessuna", la togliamo perché non vogliamo consigliare "Nessuna" se è approvato
df_bank = df_bank[df_bank['banca_consigliata'] != 'Nessuna']

# Encoder per trasformare i nomi delle banche in numeri (0, 1, 2...)
le_bank = LabelEncoder()
y_bank = le_bank.fit_transform(df_bank['banca_consigliata'])

# Le feature sono le stesse di prima (senza la colonna target banca ovviamente)
X_bank = df_bank.drop(columns=['mutuo_approvato', 'banca_consigliata'])

# Allineiamo le colonne (assicuriamoci che X_bank abbia le stesse colonne di X_app)
# (Il codice di gestione colonne sopra ha modificato df_approval, non df_bank. Rifacciamo veloce)
if 'livello_istruzione_Laurea' in X_bank.columns:
    X_bank['livello_istruzione_Diploma_o_inferiore'] = 1 - X_bank['livello_istruzione_Laurea']
if 'tipo_lavoro_Autonomo' in X_bank.columns:
    X_bank['tipo_lavoro_Dipendente'] = 1 - X_bank['tipo_lavoro_Autonomo']

# Riordiniamo le colonne per essere identiche al modello 1
X_bank = X_bank[X_app.columns]

X_bank_scaled = scaler.transform(X_bank)  # Usiamo lo stesso scaler

print("⏳ Addestro Modello Suggerimento Banca...")
model_bank = RandomForestClassifier(max_depth=7, n_estimators=100, random_state=42)
model_bank.fit(X_bank_scaled, y_bank)

# 3. Salva tutto
joblib.dump(model_approval, 'modello_mutui.pkl')
joblib.dump(model_bank, 'modello_banche.pkl')
joblib.dump(scaler, 'scaler_mutui.pkl')
joblib.dump(X_app.columns, 'colonne_mutui.pkl')
joblib.dump(le_bank, 'encoder_banche.pkl')  # Salviamo anche il traduttore dei nomi banche

print("🎉 TUTTO FATTO! Modelli salvati.")