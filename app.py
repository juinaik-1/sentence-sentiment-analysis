from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Home route to display the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle emotion detection
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    text_vector = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_vector)[0]
    return jsonify({'emotion': prediction})

if __name__ == '__main__':
    app.run(debug=True)
