from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from datetime import datetime
from database import Base



class SurveyResponseDB(Base):
    """
    Table to store survey responses and predictions
    """
    __tablename__ = "survey_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String, index=True, nullable=False)
    answers = Column(JSON, nullable=False)  # Store all survey answers as JSON
    predicted_risk = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<SurveyResponse(id={self.id}, company={self.company_name}, risk={self.predicted_risk})>"