# SensorShield 🛡️ | Industrial Anomaly Detection API

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

SensorShield is a production-ready RESTful API designed to detect anomalies in industrial time-series data (e.g., sensor telemetry, IoT devices). 
It leverages a Deep Learning Autoencoder built with PyTorch, served via FastAPI, and uses PostgreSQL to log events and prediction scores.

---

## 🧠 The Approach: Why Unsupervised Learning?

In real-world industrial scenarios, machinery operates normally 99% of the time. When a fault occurs, its signature is often unpredictable. 
A standard Supervised classification model would fail to detect an anomaly type it hasn't explicitly seen during training.

SensorShield solves this using a **Self-Supervised Autoencoder**:
1. **Training:** The PyTorch model is trained *exclusively* on healthy/normal data. It learns to compress (encode) and perfectly reconstruct (decode) normal sensor waves.
2. **Inference:** When a new wave arrives, the model tries to reconstruct it. 
3. **Detection:** If the data is normal, the Mean Squared Error (MSE) is close to 0. If it's an unforeseen anomaly, the model fails to reconstruct it, resulting in a huge MSE spike that breaches a calculated 95th-percentile threshold.

## 🏗️ Architecture

```text
[ Industrial Sensor ] 
         │ (JSON Payload - 140 features)
         ▼
 ┌───────────────┐        ┌───────────────────────┐
 │   FastAPI     │ ──────►│ PyTorch Autoencoder   │
 │ (Pydantic In) │ ◄──────│ (Reconstruction MSE)  │
 └───────┬───────┘        └───────────────────────┘
         │
         ▼ (SQLAlchemy ORM)
 ┌───────────────┐
 │  PostgreSQL   │ (Logs: timestamp, mse_score, is_anomaly)
 └───────────────┘
```

## 🚀 Quickstart (Docker)

The entire infrastructure is containerized. You do not need to install Python or PyTorch locally to run the API.

1. Clone the repository:
   ```bash
   git clone https://github.com/claudioscali/SensorShield.git
   cd SensorShield
   ```
2. Start the API and the Database:
   ```bash
   docker-compose up --build
   ```
3. The API will be available at `http://localhost:8000`. You can explore the auto-generated Swagger UI documentation at `http://localhost:8000/docs`.

## 🧪 Testing the API

To simulate a real industrial sensor sending both a healthy and a faulty signal, open a new terminal and run the included test script (requires `pandas` and `requests`):

```bash
python test_api.py
```

**Expected Output:**
```json
--- TEST 1: ONDA NORMALE ---
Status Code: 200
Risposta: {'id': 1, 'timestamp': '...', 'is_anomaly': False, 'reconstruction_error': 0.017}

--- TEST 2: ONDA ANOMALA ---
Status Code: 200
Risposta: {'id': 2, 'timestamp': '...', 'is_anomaly': True, 'reconstruction_error': 0.798}
```

## 📂 Project Structure

*   `model.py`: PyTorch Autoencoder Neural Network architecture.
*   `train.py`: ML Pipeline for training the model and calculating the threshold.
*   `main.py`: FastAPI application and business logic.
*   `database.py` & `models.py`: SQLAlchemy configuration and PostgreSQL schema.
*   `schemas.py`: Pydantic models for strict I/O validation.
*   `Dockerfile` & `docker-compose.yml`: Container orchestration setup.

---
*Developed as a portfolio project to demonstrate end-to-end Data Science and Software Engineering integration.*