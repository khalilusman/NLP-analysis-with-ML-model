#simple chatbot options code till 299



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
# client = OpenAI(api_key="")  

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























































#code for fully NLP and asking question in chat till 725


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, List, Optional, Any
# import pickle
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# # NLP Libraries
# from textblob import TextBlob
# from transformers import pipeline
# import spacy

# # Database imports
# from sqlalchemy.orm import Session
# from database import SessionLocal, engine
# import models
# import crud
# from questions_config import NLP_QUESTIONS  # must exist

# # Environment / API clients
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# # Initialize NLP Models

# print("🤖 Loading NLP models...")

# # 1) Sentiment
# sentiment_analyzer = pipeline(
#     "sentiment-analysis",
#     model="distilbert-base-uncased-finetuned-sst-2-english"
# )

# # 2) NER
# try:
#     nlp = spacy.load("en_core_web_sm")
# except Exception:
#     print("⚠️ spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
#     nlp = None

# # 3) Zero-shot risk classifier
# risk_classifier = pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli"
# )

# print(" NLP models loaded")


# # DB

# models.Base.metadata.create_all(bind=engine)


# # Load Trained ML Model

# MODEL_PATH = "../saved_models/risk_model.pkl"
# COLUMNS_PATH = "../saved_models/train_columns.pkl"

# model = None
# train_columns: Optional[List[str]] = None

# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     with open(COLUMNS_PATH, "rb") as f:
#         train_columns = pickle.load(f)
#     print("✓ Traditional ML model loaded")
# except Exception as e:
#     print(f"⚠️ ML model not loaded: {e}")
#     model = None
#     train_columns = None


# # FastAPI

# app = FastAPI(
#     title="NLP-Powered Risk Assessment API",
#     description="Full NLP + Trained-Model analysis for business risk prediction",
#     version="3.1.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # NLP helpers

# def analyze_sentiment(text: str) -> Dict[str, Any]:
#     """Analyze sentiment of text responses"""
#     try:
#         result = sentiment_analyzer(text[:512])[0]
#         blob = TextBlob(text)
#         return {
#             "label": result.get("label", "NEUTRAL"),
#             "score": float(result.get("score", 0.5)),
#             "polarity": float(blob.sentiment.polarity),
#             "subjectivity": float(blob.sentiment.subjectivity),
#         }
#     except Exception as e:
#         print(f"Sentiment analysis error: {e}")
#         return {"label": "NEUTRAL", "score": 0.5, "polarity": 0.0, "subjectivity": 0.5}

# def extract_entities(text: str) -> List[Dict[str, Any]]:
#     """Extract named entities"""
#     if not nlp:
#         return []
#     doc = nlp(text)
#     entities = []
#     for ent in doc.ents:
#         entities.append({
#             "text": ent.text,
#             "label": ent.label_,
#             "description": spacy.explain(ent.label_) or "",
#         })
#     return entities

# def extract_keywords(text: str) -> List[str]:
#     """Extract key phrases"""
#     try:
#         blob = TextBlob(text)
#         # De-duplicate and cap at 10
#         return list(dict.fromkeys(map(str, blob.noun_phrases)))[:10]
#     except Exception:
#         return []

# def classify_risk_from_text(text: str, candidate_labels: List[str]) -> Optional[Dict[str, Any]]:
#     """Zero-shot risk classification"""
#     try:
#         result = risk_classifier(
#             text[:1024],
#             candidate_labels,
#             multi_label=False
#         )
#         # HF returns scores aligned with labels
#         return {
#             "predicted_risk": result["labels"][0],
#             "confidence": float(result["scores"][0]),
#             "all_scores": {lbl: float(scr) for lbl, scr in zip(result["labels"], result["scores"])},
#             "engine": "zero-shot-bart-large-mnli"
#         }
#     except Exception as e:
#         print(f"Risk classification error: {e}")
#         return None

# def nlp_risk_prediction(text_responses: Dict[str, str], structured_data: Dict[str, Any]) -> Dict[str, Any]:
#     """Main NLP prediction function"""
#     risk_categories = [
#         "Financial risk",
#         "Operational risk",
#         "Strategic risk",
#         "Technical risk",
#         "Compliance risk",
#         "Project churn",
#         "Information security"
#     ]

#     # Concatenate all text
#     all_text = " ".join(
#         f"{v}" for _, v in text_responses.items() if isinstance(v, str) and v.strip()
#     )
#     print(f"📝 Combined text length: {len(all_text)} chars")

#     nlp_prediction = None
#     sentiment = None
#     keywords: List[str] = []

#     if all_text.strip():
#         nlp_prediction = classify_risk_from_text(all_text, risk_categories)
#         sentiment = analyze_sentiment(all_text)
#         keywords = extract_keywords(all_text)

#     return {
#         "nlp_prediction": nlp_prediction,
#         "sentiment_analysis": sentiment or {},
#         "risk_keywords": keywords,
#         "text_length": len(all_text)
#     }


# # Trained ML helpers

# def _build_feature_row(structured_data: Dict[str, Any], columns: List[str]) -> pd.DataFrame:
#     """
#     Create a single-row DataFrame aligned to training columns.
#     - Extract values from structured_data by exact key match.
#     - If a key is missing, fill with NaN (let pipeline handle imputation/encoders).
#     """
#     row = {col: structured_data.get(col, np.nan) for col in columns}
#     X = pd.DataFrame([row], columns=columns)
#     return X

# def predict_with_traditional_model(structured_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#     """
#     Predict using the saved/trained model.
#     Supports:
#       - A bare estimator with separate preprocessing handled at training (pickle a Pipeline ideally)
#       - A full sklearn Pipeline (recommended)
#     """
#     if model is None or train_columns is None:
#         return None
#     try:
#         X = _build_feature_row(structured_data, train_columns)

#         # If it's a pipeline, transformation happens inside.
#         # Prefer predict_proba if available
#         if hasattr(model, "predict_proba"):
#             proba = model.predict_proba(X)[0]
#             classes = list(getattr(model, "classes_", [str(i) for i in range(len(proba))]))
#             top_idx = int(np.argmax(proba))
#             return {
#                 "predicted_risk": str(classes[top_idx]),
#                 "confidence": float(proba[top_idx]),
#                 "all_scores": {str(c): float(p) for c, p in zip(classes, proba)},
#                 "engine": "trained-ml-proba"
#             }
#         else:
#             y = model.predict(X)[0]
#             # No probability => conservative confidence
#             return {
#                 "predicted_risk": str(y),
#                 "confidence": 0.51,
#                 "all_scores": {},
#                 "engine": "trained-ml"
#             }
#     except Exception as e:
#         print(f"Traditional model prediction error: {e}")
#         return None


# # Advice generation

# def generate_nlp_advice(risk_type: str, sentiment: Dict[str, Any], keywords: List[str]) -> str:
#     """Generate AI advice text (OpenAI). Falls back to static if key missing."""
#     sentiment_desc = "concerned" if sentiment and float(sentiment.get("polarity", 0)) < 0 else "optimistic"
#     keywords_str = ", ".join(keywords[:5]) if keywords else "general concerns"

#     prompt = f"""As a risk management expert, provide specific advice for a software company facing {risk_type}.

# Context from their responses:
# - Overall sentiment: {sentiment_desc}
# - Key concerns mentioned: {keywords_str}

# Provide 3-4 actionable recommendations."""
#     try:
#         if not client:
#             raise RuntimeError("OpenAI API key missing")
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an expert risk management consultant."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=300,
#             temperature=0.7
#         )
#         return (response.choices[0].message.content or "").strip()
#     except Exception as e:
#         print(f"OpenAI error: {e}")
#         return f"- Establish a formal {risk_type} register and monitoring cadence.\n- Define triggers with owners and SLAs.\n- Run scenario tests and tabletop exercises quarterly.\n- Tighten controls, logging, and post-incident reviews with clear KPIs."


# # Schemas

# class NLPPredictionRequest(BaseModel):
#     company_name: str
#     text_responses: Dict[str, str]
#     structured_data: Dict[str, Any]

# class NLPPredictionResponse(BaseModel):
#     predicted_risk: str
#     confidence: float
#     source_engine: str
#     nlp_analysis: Dict[str, Any]
#     sentiment_analysis: Dict[str, Any]
#     risk_keywords: List[str]
#     advice: str
#     timestamp: str


# # Routes

# @app.get("/")
# async def root():
#     return {
#         "status": "online",
#         "message": "NLP-Powered Risk Assessment API",
#         "version": "3.1.0",
#         "nlp_enabled": True,
#         "models_loaded": {
#             "sentiment_analysis": True,
#             "entity_recognition": nlp is not None,
#             "risk_classification": True,
#             "traditional_ml": model is not None
#         }
#     }

# @app.get("/questions")
# async def get_questions():
#     """Return NLP questions for frontend"""
#     return {
#         "total_questions": len(NLP_QUESTIONS),
#         "questions": NLP_QUESTIONS,
#         "nlp_enabled": True
#     }

# @app.post("/predict-nlp", response_model=NLPPredictionResponse)
# async def predict_with_nlp(request: NLPPredictionRequest):
#     """
#     Hybrid prediction endpoint:
#     - NLP zero-shot classification on concatenated text
#     - Trained ML model on structured_data
#     - Select higher-confidence result (if ML has proba).
#       If ML has no proba, it wins only if NLP is missing.
#     """
#     try:
#         print("\n=== 🚀 HYBRID PREDICTION ===")
#         print(f"Company: {request.company_name}")
#         print(f"Text responses: {len(request.text_responses)} keys")
#         print(f"Structured data: {len(request.structured_data)} keys")

#         # 1) NLP analysis (zero-shot + sentiment + keywords)
#         nlp_results = nlp_risk_prediction(request.text_responses, request.structured_data)
#         zshot = nlp_results.get("nlp_prediction")  # may be None

#         # 2) Trained model prediction
#         ml_results = predict_with_traditional_model(request.structured_data)

#         # 3) Selection / Ensembling
#         chosen_engine = "fallback"
#         if ml_results and "confidence" in ml_results and ml_results["confidence"] is not None and \
#            zshot and "confidence" in zshot and zshot["confidence"] is not None:
#             # Both available with confidence: choose higher
#             if float(ml_results["confidence"]) >= float(zshot["confidence"]):
#                 chosen = ml_results
#             else:
#                 chosen = zshot
#         elif ml_results:
#             # ML only (could be no proba, but we still use it if NLP missing)
#             chosen = ml_results
#         elif zshot:
#             chosen = zshot
#         else:
#             chosen = {"predicted_risk": "Strategic risk", "confidence": 0.5, "engine": "default"}

#         predicted_risk = str(chosen.get("predicted_risk", "Strategic risk"))
#         confidence = float(chosen.get("confidence", 0.5))
#         chosen_engine = str(chosen.get("engine", "unknown"))

#         print(f"🧠 Selected: {predicted_risk} ({confidence:.2%}) via {chosen_engine}")

#         # 4) Advice
#         advice = generate_nlp_advice(
#             predicted_risk,
#             nlp_results.get("sentiment_analysis", {}),
#             nlp_results.get("risk_keywords", [])
#         )

#         # 5) Persist
#         db = SessionLocal()
#         try:
#             crud.create_survey_response(
#                 db=db,
#                 company_name=request.company_name,
#                 answers={**request.text_responses, **request.structured_data},
#                 predicted_risk=predicted_risk,
#                 confidence=confidence
#             )
#             print("💾 Saved to database")
#         except Exception as db_error:
#             print(f"⚠️ Database save failed: {db_error}")
#         finally:
#             db.close()

#         # 6) Respond
#         return NLPPredictionResponse(
#             predicted_risk=predicted_risk,
#             confidence=confidence,
#             source_engine=chosen_engine,
#             nlp_analysis=zshot or {},
#             sentiment_analysis=nlp_results.get("sentiment_analysis", {}),
#             risk_keywords=nlp_results.get("risk_keywords", []),
#             advice=advice,
#             timestamp=datetime.now().isoformat()
#         )

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/analyze-text")
# async def analyze_text_endpoint(text: str):
#     """Test NLP analysis utilities"""
#     return {
#         "sentiment": analyze_sentiment(text),
#         "keywords": extract_keywords(text),
#         "entities": extract_entities(text)
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



























from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn
import os
from openai import OpenAI
from dotenv import load_dotenv
from column_mapper import preprocess_chatbot_answers

# Database imports
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import crud

# Questions config
from questions_config import QUESTIONS, get_survey_questions, get_metadata_questions

# Initialize OpenAI client
client = OpenAI(api_key="")  

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Risk Prediction Chatbot API",
    description="Chatbot API for predicting business risks using AI/ML",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Feature Columns
MODEL_PATH = "../saved_models/risk_model.pkl"
COLUMNS_PATH = "../saved_models/train_columns.pkl"
LABEL_ENCODER_PATH = "../saved_models/label_encoder.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

try:
    with open(COLUMNS_PATH, 'rb') as f:
        train_columns = pickle.load(f)
    print(f"✓ Feature columns loaded: {len(train_columns)} columns")
except Exception as e:
    print(f"✗ Error loading columns: {e}")
    train_columns = None

try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder loaded: {len(label_encoder.classes_)} classes")
except Exception as e:
    print(f" Label encoder not found (using RandomForest): {e}")
    label_encoder = None

# Pydantic Models
class ChatMessage(BaseModel):
    question_id: str
    answer: Any

class ChatSession(BaseModel):
    company_name: Optional[str] = None
    answers: Dict[str, Any] = {}
    
class PredictionRequest(BaseModel):
    company_name: str
    answers: Dict[str, Any]
    custom_risks: Optional[List[str]] = []  # NEW: Add custom risks field
    
class CustomRiskAdvice(BaseModel):
    risk_name: str
    advice: str
    
class PredictionResponse(BaseModel):
    predicted_risk: str
    confidence: float
    risk_probabilities: Dict[str, float]
    advice: str
    custom_risk_advice: Optional[List[CustomRiskAdvice]] = []  # NEW: Custom risk advice
    timestamp: str


def get_risk_advice_from_ai(risk_type: str) -> str:
    """
    Get AI-generated advice from OpenAI based on predicted risk
    """
    try:
        prompt = f"How can we overcome {risk_type} risk in a software company? Provide practical, actionable advice in 3-4 sentences."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert business risk management consultant specializing in software companies. Provide concise, actionable advice."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        advice = response.choices[0].message.content.strip()
        return advice
        
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return f"To mitigate {risk_type}, implement comprehensive risk management practices, conduct regular assessments, and consult with industry experts for tailored solutions."


def get_custom_risk_advice(risk_name: str) -> str:
    """
    Get AI-generated advice for custom risks entered by users
    """
    try:
        prompt = f"A software company is concerned about '{risk_name}' risk. Provide practical, actionable advice on how to identify, prevent, and mitigate this risk in 3-4 sentences."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert business risk management consultant. Analyze the given risk and provide specific, actionable mitigation strategies."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=250,
            temperature=0.7
        )
        
        advice = response.choices[0].message.content.strip()
        return advice
        
    except Exception as e:
        print(f"OpenAI API Error for custom risk '{risk_name}': {e}")
        return f"For {risk_name}, conduct thorough risk assessment, implement monitoring systems, develop mitigation strategies, and regularly review your risk management plan."


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Enterprise Risk Prediction Chatbot API",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "total_questions": len(QUESTIONS)
    }

@app.get("/questions")
async def get_questions():
    """Get all survey questions for chatbot"""
    return {
        "total_questions": len(QUESTIONS),
        "questions": QUESTIONS
    }

@app.get("/questions/{question_id}")
async def get_question(question_id: str):
    """Get a specific question by ID"""
    question = None
    for q in QUESTIONS:
        if q["id"] == question_id:
            question = q
            break
    
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    return question

@app.post("/predict")
async def predict_risk(request: PredictionRequest):
    """Predict risk type based on survey responses"""
    if model is None or train_columns is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        print(f"\n=== Prediction Request ===")
        print(f"Company: {request.company_name}")
        print(f"Answers received: {len(request.answers)} questions")
        print(f"Custom risks: {len(request.custom_risks)} - {request.custom_risks}")
        
        # Preprocess survey data using the mapper
        X = preprocess_chatbot_answers(request.answers, train_columns)
        print(f"Preprocessed shape: {X.shape}")
        print(f"Non-zero columns: {(X != 0).sum().sum()}")
        
        # Make prediction
        prediction_encoded = int(model.predict(X)[0])
        print(f"Prediction (encoded): {prediction_encoded}")
        
        # Decode prediction if using XGBoost with label encoder
        if label_encoder is not None:
            prediction = str(label_encoder.inverse_transform([prediction_encoded])[0])
            risk_classes = label_encoder.classes_
        else:
            prediction = str(prediction_encoded)
            risk_classes = model.classes_
        
        print(f"Prediction (decoded): {prediction}")
        
        probabilities = model.predict_proba(X)[0]
        print(f"Probabilities shape: {probabilities.shape}")
        
        # Create probability dictionary
        risk_probs = {
            str(risk_classes[i]): float(probabilities[i]) 
            for i in range(len(risk_classes))
        }
        
        # Get confidence
        confidence = float(max(probabilities))
        print(f"Confidence: {confidence}")
        
        # Get AI-generated advice for predicted risk
        advice = get_risk_advice_from_ai(prediction)
        print(f"Main advice generated: {len(advice)} chars")
        
        # NEW: Get advice for custom risks
        custom_risk_advice = []
        if request.custom_risks:
            print(f"Generating advice for {len(request.custom_risks)} custom risks...")
            for risk_name in request.custom_risks:
                risk_advice = get_custom_risk_advice(risk_name)
                custom_risk_advice.append(
                    CustomRiskAdvice(risk_name=risk_name, advice=risk_advice)
                )
                print(f"  ✓ Advice for '{risk_name}': {len(risk_advice)} chars")
        
        response = PredictionResponse(
            predicted_risk=prediction,
            confidence=confidence,
            risk_probabilities=risk_probs,
            advice=advice,
            custom_risk_advice=custom_risk_advice,
            timestamp=datetime.now().isoformat()
        )
        
        # Save to database
        db = SessionLocal()
        try:
            # Include custom risks in answers dict for storage
            answers_with_custom = request.answers.copy()
            if request.custom_risks:
                answers_with_custom['custom_risks'] = request.custom_risks
            
            crud.create_survey_response(
                db=db,
                company_name=request.company_name,
                answers=answers_with_custom,  # Save with custom risks
                predicted_risk=prediction,
                confidence=confidence
            )
            print("Saved to database successfully")
        except Exception as db_error:
            print(f"Warning: Failed to save to database: {db_error}")
        finally:
            db.close()
        
        print(f"=== Prediction Complete ===\n")
        return response
        
    except Exception as e:
        print(f"\n!!! ERROR in prediction !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/history/{company_name}")
async def get_company_history(company_name: str):
    """Retrieve historical predictions for a company"""
    db = SessionLocal()
    try:
        history = crud.get_company_responses(db, company_name)
        
        return {
            "company_name": company_name,
            "total_responses": len(history),
            "responses": [
                {
                    "id": record.id,
                    "predicted_risk": record.predicted_risk,
                    "confidence": record.confidence,
                    "timestamp": record.timestamp.isoformat()
                }
                for record in history
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        db.close()

@app.get("/statistics")
async def get_statistics():
    """Get overall statistics from database"""
    db = SessionLocal()
    try:
        stats = crud.get_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )