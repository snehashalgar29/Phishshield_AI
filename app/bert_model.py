import os
from transformers import pipeline
MODEL_DIR = 'models/bert_phish'
bert_pipeline = None
if os.path.exists(MODEL_DIR):
    try:
        bert_pipeline = pipeline('text-classification', model=MODEL_DIR, tokenizer=MODEL_DIR, return_all_scores=True)
    except Exception:
        bert_pipeline = None
