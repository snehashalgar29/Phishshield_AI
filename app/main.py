from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.text_preprocess import clean_text
from app.url_checks import analyze_url
from app.bert_model import bert_pipeline
import joblib, os

app = FastAPI(title='PhishShield AI - Full Run Demo')

class TextIn(BaseModel):
    text: str

class URLIn(BaseModel):
    url: str

# Load demo model if BERT not available
demo_model = None
demo_model_path = 'models/demo_model.joblib'
if os.path.exists(demo_model_path):
    demo_model = joblib.load(demo_model_path)

@app.get('/')
def root():
    return {'status':'running', 'bert_loaded': bert_pipeline is not None, 'demo_loaded': demo_model is not None}

@app.post('/analyze_text')
def analyze_text(payload: TextIn):
    t = clean_text(payload.text)
    if bert_pipeline is not None:
        preds = bert_pipeline(t)[0]
        # find phish label score (LABEL_1 assumed phish)
        phish_score = 0.0
        for p in preds:
            if p['label'] in ('LABEL_1','PHISH','1'):
                phish_score = p['score']
        score = int(phish_score * 100)
        classification = 'malicious' if score >= 70 else 'suspicious' if score >= 40 else 'safe'
        return {'model':'bert','phish_probability':phish_score,'risk_score':score,'classification':classification,'raw':preds}
    elif demo_model is not None:
        prob = demo_model.predict_proba([t])[0][1]
        score = int(prob * 100)
        classification = 'malicious' if score >= 70 else 'suspicious' if score >= 40 else 'safe'
        return {'model':'demo','phish_probability':prob,'risk_score':score,'classification':classification}
    else:
        raise HTTPException(status_code=503, detail='No model available. Please run training to create models/demo_model.joblib or place a BERT model in models/bert_phish')

@app.post('/analyze_url')
def analyze_url_endpoint(payload: URLIn):
    return analyze_url(payload.url)
