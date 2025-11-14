# Light-weight demo trainer: trains TF-IDF + LogisticRegression on the sample CSV and saves model to models/demo_model.joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib, os

os.makedirs('models', exist_ok=True)
df = pd.read_csv('data/phish_dataset.csv')
X = df['text']
y = df['label']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X,y)
joblib.dump(pipeline, 'models/demo_model.joblib')
print('Saved demo model to models/demo_model.joblib')
