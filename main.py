import json
import torch
import joblib
import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

# Importiamo i file creati
from database import engine, get_db
import models, schemas
from model import AnomalyAutoencoder

# classe SensorLog in SQL (CREATE TABLE ...)
models.Base.metadata.create_all(bind=engine) # Dice a SQLAlchemy di creare le tabelle in PostgreSQL

print("Caricamento modello, scaler e soglia...")

# Inizializza l'architettura vuota e "inietta" i pesi addestrati
ml_model = AnomalyAutoencoder(input_dim=140)
ml_model.load_state_dict(torch.load("model.pth"))
ml_model.eval() # Modalità inferenza (disabilita calcoli extra)

# Carica lo scaler per normalizzare i nuovi dati in ingresso
scaler = joblib.load("data/scaler.save")

# Carica la soglia di tolleranza calcolata nel Giorno 1
with open("threshold.json", "r") as f:
    threshold_data = json.load(f)
    ANOMALY_THRESHOLD = threshold_data["anomaly_threshold"]

# 3. Inizializzazione dell'app FastAPI
app = FastAPI(title="SensorShield API", description="API per Anomaly Detection Industriale")

# 4. L'Endpoint principale
@app.post("/predict", response_model=schemas.SensorOutput)
def predict_anomaly(payload: schemas.SensorInput, db: Session = Depends(get_db)):
    try:
        # A. PREPARAZIONE DATI
        raw_data = np.array(payload.features).reshape(1, -1) # trasforma la lista in una riga singola di una matrice con 1 riga e 140 colonne
        
        # Riapplichiamo la stessa normalizzazione usata in addestramento (schiacciandoli tra 0 e 1)
        scaled_data = scaler.transform(raw_data)
        
        # Convertiamo il dato scalato nel formato compreso dalla rete neurale
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
        
        # B. INFERENZA CON L'AUTOENCODER
        with torch.no_grad(): # Nessun calcolo delle derivate = massima velocità e minor RAM
            reconstruction = ml_model(tensor_data)
            # Calcoliamo l'errore quadratico medio della singola onda
            mse = torch.mean((tensor_data - reconstruction)**2).item()
            
        # C. DECISIONE
        is_anomaly = bool(mse > ANOMALY_THRESHOLD)
        
        # D. SALVATAGGIO SU DATABASE
        # Creiamo un nuovo record per la tabella SensorLog
        db_log = models.SensorLog(
            sensor_data=json.dumps(payload.features), # Convertiamo l'array originale in una stringa di testo, per salvarlo in una singola cella del db
            reconstruction_error=mse,
            is_anomaly=is_anomaly
        )
        
        db.add(db_log)  
        db.commit()      
        db.refresh(db_log)# Aggiorna l'oggetto (così il DB ci restituisce l'ID autoincrementale appena creato)
        
        # Restituiamo il log. FastAPI lo convertirà automaticamente nel formato JSON di schemas.SensorOutput
        return db_log
        
    except Exception as e:
        # In caso di errore (es. database irraggiungibile), restituiamo un HTTP 500 
        raise HTTPException(status_code=500, detail=str(e))