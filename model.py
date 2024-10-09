import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import nltk
import re

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Load the dataset
data = pd.read_csv('dataset/tweet_emotions.csv')
data = data[['content', 'sentiment']]

# Preprocess the dataset
data['content'] = data['content'].apply(preprocess_text)

# Split the dataset
X = data['content']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Train a logistic regression model for multi-class classification
model = LogisticRegression(multi_class='ovr', max_iter=200)
model.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')