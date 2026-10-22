from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
# Pydantic per parlare con il Client/Utente: questo invia JSON all'API

# 1. Schema IN ENTRATA: Cosa ci invia il sensore/utente tramite POST
class SensorInput(BaseModel):
    # Diciamo all'API di aspettarsi una lista di 140 numeri float. 
    features: List[float] = Field(..., min_length=140, max_length=140, description="140 misurazioni del sensore")

# 2. Schema IN USCITA: Cosa risponde la nostra API al client
class SensorOutput(BaseModel):
    id: int
    timestamp: datetime
    is_anomaly: bool
    reconstruction_error: float

    # Configurazione speciale per dire a Pydantic di saper leggere i dati da un oggetto SQLAlchemy
    class Config:
        from_attributes = True