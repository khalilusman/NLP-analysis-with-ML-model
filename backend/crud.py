from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, List
import models
from datetime import datetime


def create_survey_response(
    db: Session, 
    company_name: str, 
    answers: Dict, 
    predicted_risk: str, 
    confidence: float
) -> models.SurveyResponseDB:
    """
    Create new survey response record
    """
    db_response = models.SurveyResponseDB(
        company_name=company_name,
        answers=answers,
        predicted_risk=predicted_risk,
        confidence=confidence,
        timestamp=datetime.utcnow()
    )
    db.add(db_response)
    db.commit()
    db.refresh(db_response)
    return db_response

def get_survey_response(db: Session, response_id: int) -> models.SurveyResponseDB:
    """
    Get survey response by ID
    """
    return db.query(models.SurveyResponseDB).filter(
        models.SurveyResponseDB.id == response_id
    ).first()

def get_company_responses(db: Session, company_name: str) -> List[models.SurveyResponseDB]:
    """
    Get all responses for a specific company
    """
    return db.query(models.SurveyResponseDB).filter(
        models.SurveyResponseDB.company_name == company_name
    ).order_by(models.SurveyResponseDB.timestamp.desc()).all()

def get_all_responses(db: Session, skip: int = 0, limit: int = 100) -> List[models.SurveyResponseDB]:
    """
    Get all survey responses with pagination
    """
    return db.query(models.SurveyResponseDB).offset(skip).limit(limit).all()

def get_statistics(db: Session) -> Dict:
    """
    Get overall statistics from database
    """
    total_responses = db.query(models.SurveyResponseDB).count()
    
    # Count responses by risk type
    risk_distribution = db.query(
        models.SurveyResponseDB.predicted_risk,
        func.count(models.SurveyResponseDB.id)
    ).group_by(models.SurveyResponseDB.predicted_risk).all()
    
    # Count unique companies
    unique_companies = db.query(
        func.count(func.distinct(models.SurveyResponseDB.company_name))
    ).scalar()
    
    # Average confidence
    avg_confidence = db.query(
        func.avg(models.SurveyResponseDB.confidence)
    ).scalar()
    
    return {
        "total_responses": total_responses,
        "unique_companies": unique_companies,
        "average_confidence": float(avg_confidence) if avg_confidence else 0.0,
        "risk_distribution": {risk: count for risk, count in risk_distribution}
    }

def delete_survey_response(db: Session, response_id: int) -> bool:
    """
    Delete a survey response by ID
    """
    response = db.query(models.SurveyResponseDB).filter(
        models.SurveyResponseDB.id == response_id
    ).first()
    
    if response:
        db.delete(response)
        db.commit()
        return True
    return False