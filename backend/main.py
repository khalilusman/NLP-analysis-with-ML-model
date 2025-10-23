# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, List, Optional, Any
# import pickle
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import uvicorn
# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# from column_mapper import preprocess_chatbot_answers

# # Load environment variables
# # env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# # load_dotenv(dotenv_path=env_path)

# # Database imports
# from sqlalchemy.orm import Session
# from database import SessionLocal, engine
# import models
# import crud

# # Questions config
# from questions_config import QUESTIONS, get_survey_questions, get_metadata_questions

# # Initialize OpenAI client

# # load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
# # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

# # Create database tables
# models.Base.metadata.create_all(bind=engine)

# # Initialize FastAPI app
# app = FastAPI(
#     title="Enterprise Risk Prediction Chatbot API",
#     description="Chatbot API for predicting business risks using AI/ML",
#     version="2.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load Model and Feature Columns
# MODEL_PATH = "../saved_models/risk_model.pkl"
# COLUMNS_PATH = "../saved_models/train_columns.pkl"
# LABEL_ENCODER_PATH = "../saved_models/label_encoder.pkl"

# try:
#     with open(MODEL_PATH, 'rb') as f:
#         model = pickle.load(f)
#     print("✓ Model loaded successfully")
# except Exception as e:
#     print(f"✗ Error loading model: {e}")
#     model = None

# try:
#     with open(COLUMNS_PATH, 'rb') as f:
#         train_columns = pickle.load(f)
#     print(f"✓ Feature columns loaded: {len(train_columns)} columns")
# except Exception as e:
#     print(f"✗ Error loading columns: {e}")
#     train_columns = None

# try:
#     with open(LABEL_ENCODER_PATH, 'rb') as f:
#         label_encoder = pickle.load(f)
#     print(f"✓ Label encoder loaded: {len(label_encoder.classes_)} classes")
# except Exception as e:
#     print(f" Label encoder not found (using RandomForest): {e}")
#     label_encoder = None

# # Pydantic Models
# class ChatMessage(BaseModel):
#     question_id: str
#     answer: Any  # Can be string, list, dict, etc.

# class ChatSession(BaseModel):
#     company_name: Optional[str] = None
#     answers: Dict[str, Any] = {}
    
# class PredictionRequest(BaseModel):
#     company_name: str
#     answers: Dict[str, Any]
    
# class PredictionResponse(BaseModel):
#     predicted_risk: str
#     confidence: float
#     risk_probabilities: Dict[str, float]
#     advice: str
#     timestamp: str


# def get_risk_advice_from_ai(risk_type: str) -> str:
#     """
#     Get AI-generated advice from OpenAI based on predicted risk
#     """
#     try:
#         prompt = f"How can we overcome {risk_type} risk in a software company? Provide practical, actionable advice in 3-4 sentences."
        
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",  # or "gpt-4" for better results
#             messages=[
#                 {
#                     "role": "system", 
#                     "content": "You are an expert business risk management consultant specializing in software companies. Provide concise, actionable advice."
#                 },
#                 {
#                     "role": "user", 
#                     "content": prompt
#                 }
#             ],
#             max_tokens=200,
#             temperature=0.7
#         )
        
#         advice = response.choices[0].message.content.strip()
#         return advice
        
#     except Exception as e:
#         print(f"OpenAI API Error: {e}")
#         # Fallback advice if OpenAI fails
#         return f"To mitigate {risk_type}, implement comprehensive risk management practices, conduct regular assessments, and consult with industry experts for tailored solutions."

# # API Endpoints

# @app.get("/")
# async def root():
#     """Health check endpoint"""
#     return {
#         "status": "online",
#         "message": "Enterprise Risk Prediction Chatbot API",
#         "version": "2.0.0",
#         "model_loaded": model is not None,
#         "total_questions": len(QUESTIONS)
#     }

# @app.get("/questions")
# async def get_questions():
#     """Get all survey questions for chatbot"""
#     return {
#         "total_questions": len(QUESTIONS),
#         "questions": QUESTIONS
#     }

# @app.get("/questions/{question_id}")
# async def get_question(question_id: str):
#     """Get a specific question by ID"""
#     question = None
#     for q in QUESTIONS:
#         if q["id"] == question_id:
#             question = q
#             break
    
#     if not question:
#         raise HTTPException(status_code=404, detail="Question not found")
    
#     return question

# @app.post("/predict")
# async def predict_risk(request: PredictionRequest):
#     """Predict risk type based on survey responses"""
#     if model is None or train_columns is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
    
#     try:
#         print(f"\n=== Prediction Request ===")
#         print(f"Company: {request.company_name}")
#         print(f"Answers received: {len(request.answers)} questions")
        
#         # Preprocess survey data using the mapper
#         X = preprocess_chatbot_answers(request.answers, train_columns)
#         print(f"Preprocessed shape: {X.shape}")
#         print(f"Non-zero columns: {(X != 0).sum().sum()}")
        
#         # Make prediction
#         prediction_encoded = int(model.predict(X)[0])  # Convert numpy int to Python int
#         print(f"Prediction (encoded): {prediction_encoded}")
        
#         # Decode prediction if using XGBoost with label encoder
#         if label_encoder is not None:
#             prediction = str(label_encoder.inverse_transform([prediction_encoded])[0])
#             risk_classes = label_encoder.classes_
#         else:
#             prediction = str(prediction_encoded)
#             risk_classes = model.classes_
        
#         print(f"Prediction (decoded): {prediction}")
        
#         probabilities = model.predict_proba(X)[0]
#         print(f"Probabilities shape: {probabilities.shape}")
#         print(f"Risk classes: {risk_classes}")
        
#         # Create probability dictionary with STRING keys
#         risk_probs = {
#             str(risk_classes[i]): float(probabilities[i]) 
#             for i in range(len(risk_classes))
#         }
        
#         # Get confidence
#         confidence = float(max(probabilities))
#         print(f"Confidence: {confidence}")
        
#         # Get AI-generated advice
#         advice = get_risk_advice_from_ai(prediction)
#         print(f"Advice generated: {len(advice)} chars")
        
#         response = PredictionResponse(
#             predicted_risk=prediction,
#             confidence=confidence,
#             risk_probabilities=risk_probs,
#             advice=advice,
#             timestamp=datetime.now().isoformat()
#         )
        
#         # Save to database
#         db = SessionLocal()
#         try:
#             crud.create_survey_response(
#                 db=db,
#                 company_name=request.company_name,
#                 answers=request.answers,
#                 predicted_risk=prediction,
#                 confidence=confidence
#             )
#             print("Saved to database successfully")
#         except Exception as db_error:
#             print(f"Warning: Failed to save to database: {db_error}")
#         finally:
#             db.close()
        
#         print(f"=== Prediction Complete ===\n")
#         return response
        
#     except Exception as e:
#         print(f"\n!!! ERROR in prediction !!!")
#         print(f"Error type: {type(e).__name__}")
#         print(f"Error message: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# @app.get("/history/{company_name}")
# async def get_company_history(company_name: str):
#     """Retrieve historical predictions for a company"""
#     db = SessionLocal()
#     try:
#         history = crud.get_company_responses(db, company_name)
        
#         return {
#             "company_name": company_name,
#             "total_responses": len(history),
#             "responses": [
#                 {
#                     "id": record.id,
#                     "predicted_risk": record.predicted_risk,
#                     "confidence": record.confidence,
#                     "timestamp": record.timestamp.isoformat()
#                 }
#                 for record in history
#             ]
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
#     finally:
#         db.close()

# @app.get("/statistics")
# async def get_statistics():
#     """Get overall statistics from database"""
#     db = SessionLocal()
#     try:
#         stats = crud.get_statistics(db)
#         return stats
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
#     finally:
#         db.close()

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )





from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv

# NLP Libraries
from textblob import TextBlob
from transformers import pipeline
import spacy

# Load environment
load_dotenv()

# Database imports
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import crud
from questions_config import NLP_QUESTIONS  # FIXED: Correct import

# Initialize NLP Models
print("🤖 Loading NLP models...")

# 1. Sentiment Analysis
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Named Entity Recognition
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️ Run: python -m spacy download en_core_web_sm")
    nlp = None

# 3. Text Classification for Risk
risk_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

print(" NLP models loaded!")

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(
    title="NLP-Powered Risk Assessment API",
    description="Full NLP analysis for business risk prediction",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # FIXED: Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Model (kept for comparison)
MODEL_PATH = "../saved_models/risk_model.pkl"
COLUMNS_PATH = "../saved_models/train_columns.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(COLUMNS_PATH, 'rb') as f:
        train_columns = pickle.load(f)
    print("✓ Traditional ML model loaded")
except Exception as e:
    print(f"⚠️ ML model not loaded: {e}")
    model = None
    train_columns = None


def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment of text responses"""
    try:
        result = sentiment_analyzer(text[:512])[0]  # Limit to 512 chars
        blob = TextBlob(text)
        
        return {
            "label": result['label'],
            "score": result['score'],
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {"label": "NEUTRAL", "score": 0.5, "polarity": 0, "subjectivity": 0.5}

def extract_entities(text: str) -> List[Dict]:
    """Extract named entities"""
    if not nlp:
        return []
    
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "description": spacy.explain(ent.label_)
        })
    
    return entities

def extract_keywords(text: str) -> List[str]:
    """Extract key phrases"""
    try:
        blob = TextBlob(text)
        return list(set(blob.noun_phrases))[:10]  # Limit to 10
    except:
        return []

def classify_risk_from_text(text: str, candidate_labels: List[str]) -> Dict:
    """Zero-shot risk classification"""
    try:
        result = risk_classifier(
            text[:1024],  # Limit text length
            candidate_labels,
            multi_label=False
        )
        
        return {
            "predicted_risk": result['labels'][0],
            "confidence": result['scores'][0],
            "all_scores": dict(zip(result['labels'], result['scores']))
        }
    except Exception as e:
        print(f"Risk classification error: {e}")
        return None

def nlp_risk_prediction(text_responses: Dict[str, str], structured_data: Dict) -> Dict:
    """Main NLP prediction function"""
    
    risk_categories = [
        "Financial risk",
        "Operational risk", 
        "Strategic risk",
        "Technical risk",
        "Compliance risk",
        "Project churn",
        "Information security"
    ]
    
    # Combine all text
    all_text = " ".join([
        f"{value}" 
        for key, value in text_responses.items() 
        if isinstance(value, str) and len(value) > 0
    ])
    
    print(f"📝 Combined text length: {len(all_text)} chars")
    
    # NLP Analysis
    nlp_prediction = None
    sentiment = None
    keywords = []
    
    if all_text.strip():
        nlp_prediction = classify_risk_from_text(all_text, risk_categories)
        sentiment = analyze_sentiment(all_text)
        keywords = extract_keywords(all_text)
    
    return {
        "nlp_prediction": nlp_prediction,
        "sentiment_analysis": sentiment,
        "risk_keywords": keywords,
        "text_length": len(all_text)
    }

def generate_nlp_advice(risk_type: str, sentiment: Dict, keywords: List[str]) -> str:
    """Generate AI advice"""
    
    sentiment_desc = "concerned" if sentiment and sentiment.get('polarity', 0) < 0 else "optimistic"
    keywords_str = ", ".join(keywords[:5]) if keywords else "general concerns"
    
    prompt = f"""As a risk management expert, provide specific advice for a software company facing {risk_type}.

Context from their responses:
- Overall sentiment: {sentiment_desc}
- Key concerns mentioned: {keywords_str}

Provide 3-4 actionable recommendations."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert risk management consultant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return f"To mitigate {risk_type}, implement comprehensive risk management practices, conduct regular assessments, and establish clear protocols for early detection and response."

# API ENDPOINTS

class NLPPredictionRequest(BaseModel):
    company_name: str
    text_responses: Dict[str, str]
    structured_data: Dict[str, Any]

class NLPPredictionResponse(BaseModel):
    predicted_risk: str
    confidence: float
    nlp_analysis: Dict
    sentiment_analysis: Dict
    risk_keywords: List[str]
    advice: str
    timestamp: str

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "NLP-Powered Risk Assessment API",
        "version": "3.0.0",
        "nlp_enabled": True,
        "models_loaded": {
            "sentiment_analysis": True,
            "entity_recognition": nlp is not None,
            "risk_classification": True,
            "traditional_ml": model is not None
        }
    }


@app.get("/questions")
async def get_questions():
    """Return NLP questions for frontend"""
    return {
        "total_questions": len(NLP_QUESTIONS),
        "questions": NLP_QUESTIONS,
        "nlp_enabled": True
    }

@app.post("/predict-nlp", response_model=NLPPredictionResponse)
async def predict_with_nlp(request: NLPPredictionRequest):
    """NLP-based prediction endpoint"""
    
    try:
        print("\n=== 🚀 NLP-BASED PREDICTION ===")
        print(f"Company: {request.company_name}")
        print(f"Text responses: {len(request.text_responses)}")
        print(f"Structured data: {len(request.structured_data)}")
        
        # 1. NLP Analysis
        nlp_results = nlp_risk_prediction(
            request.text_responses, 
            request.structured_data
        )
        
        print(f"NLP Result: {nlp_results['nlp_prediction']}")
        
        # Get predicted risk
        if nlp_results['nlp_prediction']:
            predicted_risk = nlp_results['nlp_prediction']['predicted_risk']
            confidence = nlp_results['nlp_prediction']['confidence']
        else:
            predicted_risk = "Strategic risk"  # Default fallback
            confidence = 0.5
        
        print(f" Predicted: {predicted_risk} ({confidence:.2%})")
        
        # 2. Generate advice
        advice = generate_nlp_advice(
            predicted_risk,
            nlp_results.get('sentiment_analysis'),
            nlp_results.get('risk_keywords', [])
        )
        
        # 3. Save to database
        db = SessionLocal()
        try:
            crud.create_survey_response(
                db=db,
                company_name=request.company_name,
                answers={**request.text_responses, **request.structured_data},
                predicted_risk=predicted_risk,
                confidence=confidence
            )
            print("💾 Saved to database")
        except Exception as db_error:
            print(f"⚠️ Database save failed: {db_error}")
        finally:
            db.close()
        
        return NLPPredictionResponse(
            predicted_risk=predicted_risk,
            confidence=confidence,
            nlp_analysis=nlp_results['nlp_prediction'] or {},
            sentiment_analysis=nlp_results.get('sentiment_analysis', {}),
            risk_keywords=nlp_results.get('risk_keywords', []),
            advice=advice,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text")
async def analyze_text_endpoint(text: str):
    """Test NLP analysis"""
    return {
        "sentiment": analyze_sentiment(text),
        "keywords": extract_keywords(text),
        "entities": extract_entities(text)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)