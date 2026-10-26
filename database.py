import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Stringa di connessione a PostgreSQL (creeremo il DB con Docker)
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
# motore che gestisce le connessioni al database
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# SessionLocal creerà una sessione (una transazione) per ogni chiamata API
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base è la classe genitore da cui erediteranno tutte le nostre tabelle
Base = declarative_base()

# Dependency: una funzione che fornisce una sessione del DB e la chiude in automatico a fine richiesta
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()