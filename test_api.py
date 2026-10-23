import requests
import pandas as pd

def test_api():
    print("Lettura dei dati grezzi originali dal sensore (CSV)...")
    # Leggiamo il file CSV ignorando i file .pt già processati
    df = pd.read_csv("ecg.csv", header=None)
    
    # Troviamo la prima onda normale (dove la colonna 0 è uguale a 1.0)
    # .iloc[0, 1:] prende la prima riga trovata, escludendo l'etichetta iniziale
    normal_wave = df[df[0] == 1.0].iloc[0, 1:].tolist()
    
    # Troviamo la prima onda anomala (dove la colonna 0 è uguale a 0.0)
    anomalous_wave = df[df[0] == 0.0].iloc[0, 1:].tolist()
    
    url = "http://localhost:8000/predict"

    print("\n--- TEST 1: ONDA NORMALE ---")
    response_1 = requests.post(url, json={"features": normal_wave})
    print(f"Status Code: {response_1.status_code}")
    print(f"Risposta: {response_1.json()}")

    print("\n--- TEST 2: ONDA ANOMALA ---")
    response_2 = requests.post(url, json={"features": anomalous_wave})
    print(f"Status Code: {response_2.status_code}")
    print(f"Risposta: {response_2.json()}")

if __name__ == "__main__":
    test_api()