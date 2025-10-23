from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLite Database (Local - for testing)
SQLALCHEMY_DATABASE_URL = "sqlite:///./risk_chatbot.db"

# For PostgreSQL (when deploying), uncomment and update:
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()