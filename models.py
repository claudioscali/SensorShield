from sqlalchemy import Column, Integer, Float, Boolean, DateTime, String
from datetime import datetime
from database import Base
# SQLAlchemy per parlare con il Database

class SensorLog(Base):
    __tablename__ = "sensor_logs"

    # Colonne della tabella PostgreSQL
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Salveremo l'onda originale come una stringa JSON per non creare 140 colonne diverse
    sensor_data = Column(String) 
    
    reconstruction_error = Column(Float)
    
    # index=True velocizzerà le future query del tipo "trovami tutte le anomalie"
    is_anomaly = Column(Boolean, index=True)