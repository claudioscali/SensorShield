import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Configurazione
DATA_FILE = "ecg.csv"

def generate_synthetic_data(filename):
    """
    Genera un dataset di test se non si possiede l'ECG5000 originale.
    140 feature temporali. Classe 1 = Normale, Classe 0 = Anomalia.
    """
    print(f"File {filename} non trovato. Generazione dataset sintetico in corso...")
    np.random.seed(42)
    n_samples = 5000
    n_features = 140
    
    # Dati normali: sinusoidi regolari con leggero rumore di fondo, 80% campioni normali
    x = np.linspace(0, 5 * np.pi, n_features)
    normal = np.sin(x) + np.random.normal(0, 0.1, (int(n_samples * 0.8), n_features))
    labels_normal = np.ones((len(normal), 1))
    
    # Dati anomali: sinusoidi con un picco improvviso e marcato al centro, 20% campioni anomali
    anomalous = np.sin(x) + np.random.normal(0, 0.1, (int(n_samples * 0.2), n_features))
    anomalous[:, 65:75] += 2.5  # Iniezione dell'anomalia (prende tutte le righe delle colonne dalla 65 alla 74, ogni riga è un'onda di 140 punti(istanti di tempo)=> aggiungo 2.5 in mezzo all'onda) 
    labels_anomalous = np.zeros((len(anomalous), 1))
    
    # Combina e salva come CSV senza header (formato standard ECG5000 Kaggle)
    data_normal = np.hstack([labels_normal, normal])
    data_anomalous = np.hstack([labels_anomalous, anomalous])
    
    df = pd.DataFrame(np.vstack([data_normal, data_anomalous]))
    # mescolo i dati normali e anomali e ne prendo il 100% e riassegno gli indici per il nuovo ordine
    df.sample(frac=1).reset_index(drop=True).to_csv(filename, index=False, header=False)
    print("Dataset sintetico creato!")

def main():
    if not os.path.exists(DATA_FILE):
        generate_synthetic_data(DATA_FILE)

    print("Caricamento dataset...")
    df = pd.read_csv(DATA_FILE, header=None)

    # Colonna 0: Label, Colonne 1-140: Serie temporale del sensore
    labels = df.iloc[:, 0] # tutte le righe ma solo la colonna 0 
    features = df.iloc[:, 1:].values # tutte le righe, le colonne dalla 1 in poi

    # Separiamo i dati normali (per il training) da quelli anomali
    normal_mask = (labels == 1) | (labels == 1.0) # Boolean Masking
    normal_data = features[normal_mask] # prende la tabella e la filtra con la maschera appena creata
    anomalous_data = features[~normal_mask]

    print(f"Trovati {len(normal_data)} campioni normali e {len(anomalous_data)} anomali.")

    # REGOLA D'ORO DEL ML: Il MinMaxScaler si allena (fit) SOLO sui dati normali
    # per evitare il "data leakage" (contaminazione dei dati di test)
    scaler = MinMaxScaler()
    normal_data_scaled = scaler.fit_transform(normal_data)
    anomalous_data_scaled = scaler.transform(anomalous_data)

    # Convertiamo in tensori PyTorch (Float32 è richiesto dai pesi della rete neurale)
    tensor_normal = torch.tensor(normal_data_scaled, dtype=torch.float32)
    tensor_anomalous = torch.tensor(anomalous_data_scaled, dtype=torch.float32)

    # Creiamo la cartella per l'output e salviamo
    os.makedirs("data", exist_ok=True)
    torch.save(tensor_normal, "data/normal_data.pt")
    torch.save(tensor_anomalous, "data/anomalous_data.pt")
    
    # Salviamo lo scaler. Ci servirà nell'API FastAPI per scalare i nuovi dati in entrata!
    joblib.dump(scaler, "data/scaler.save")

    print("✅ Preprocessing completato: dati e scaler salvati nella cartella 'data/'.")

if __name__ == "__main__":
    main()