import torch
import torch.nn as nn
import numpy as np
import json
from model import AnomalyAutoencoder

def main():
    print("Caricamento dati normali per l'addestramento...")
    # 1. Carichiamo solo i dati normali senza anomalie
    normal_data = torch.load("data/normal_data.pt")
    
    model = AnomalyAutoencoder(input_dim=140)
    
    # criterion: Misura l'errore. optimizer: Aggiusta i pesi per ridurre l'errore.
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    batch_size = 32
    
    print("Inizio addestramento (imparare la normalità)...")
    model.train()
    
    for epoch in range(epochs):
        # Mischiamo i dati per non farli imparare a memoria in ordine
        permutation = torch.randperm(normal_data.size()[0])
        epoch_loss = 0
        
        for i in range(0, normal_data.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch = normal_data[indices]
            
            # --- SELF-SUPERVISED LEARNING ---
            outputs = model(batch) # Passi 32 onde normali nel modello, che le comprime e le ricostruisce
            loss = criterion(outputs, batch) # Paragona l'output CON L'INPUT STESSO, non con una label!
            # ---------------------------------------------
            
            optimizer.zero_grad()
            loss.backward() # calcola derivate parziali, calcola per ogni peso in che direizone deve essere girato (aumentato o diminuito per abbassare l'errore)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoca [{epoch+1}/{epochs}], Loss: {epoch_loss/len(normal_data):.6f}")
            
    print("\nAddestramento completato!")
    
    # ---------------------------------------------------------
    # CALCOLO DELLA SOGLIA DI ANOMALIA
    # ---------------------------------------------------------
    print("Calcolo della soglia di allarme...")
    model.eval() # Mettiamo il modello in modalità "test", sola lettura (inferenza)
    with torch.no_grad():
        # Passiamo di nuovo tutti i dati normali nel modello addestrato
        reconstructed_normal = model(normal_data)
        
        # Calcoliamo l'errore (MSE) riga per riga
        mse_per_sample = torch.mean((normal_data - reconstructed_normal)**2, dim=1)
        
        # Fissiamo la soglia al 95° percentile: il 95% delle onde normali ha un errore inferiore a questo numero.
        threshold = np.percentile(mse_per_sample.numpy(), 95)
        print(f"Soglia impostata: {threshold:.6f}")
        
        # DIMOSTRAZIONE: Proviamo la rete sui dati anomali (che NON ha mai visto)
        anomalous_data = torch.load("data/anomalous_data.pt")
        reconstructed_anomalous = model(anomalous_data) # Invoca il metodo forward()
        mse_anomalous = torch.mean((anomalous_data - reconstructed_anomalous)**2, dim=1)
        
        print("\n--- RISULTATO DEL TEST ---")
        print(f"Errore medio sui dati normali: {torch.mean(mse_per_sample).item():.6f}")
        print(f"Errore medio sui dati anomali: {torch.mean(mse_anomalous).item():.6f}")

        
    torch.save(model.state_dict(), "model.pth")
    with open("threshold.json", "w") as f:
        json.dump({"anomaly_threshold": float(threshold)}, f)
        
    print("\n✅ Modello 'model.pth' e 'threshold.json' salvati con successo. Pronti per FastAPI!")

if __name__ == "__main__":
    main()