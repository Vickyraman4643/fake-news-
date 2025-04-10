from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the saved model
model = joblib.load('fake_news_model.pkl')

# Load the vectorizer
vectorization = joblib.load('vectorizer.pkl')

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Corrected this line
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    data = [news]
    vect = vectorization.transform(data).toarray()
    prediction = model.predict(vect)
    output = 'Fake News' if prediction[0] == 0 else 'True News '
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
