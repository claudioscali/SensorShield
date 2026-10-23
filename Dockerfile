# Partiamo da un'immagine Python ufficiale e leggera
FROM python:3.10-slim

# Impostiamo la cartella di lavoro dentro il container
WORKDIR /app

# Copiamo SOLO il requirements.txt per sfruttare la cache di Docker
COPY requirements.txt .

# Installiamo le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copiamo tutto il resto del codice e i file del modello (model.pth, ecc.)
COPY . .

# Diciamo al container che l'app comunicherà sulla porta 8000
EXPOSE 8000

# Il comando per accendere il server web
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]