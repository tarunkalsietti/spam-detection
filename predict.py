import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_message(message):
    # Simple text cleaning like training
    import string
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    import nltk

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def clean_text(text):
        text = text.lower()
        text = ''.join([c for c in text if c not in string.punctuation])
        tokens = text.split()
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probas = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None

    label = "SPAM" if prediction == 1 else "HAM"
    print("\n📨 Input:", message)
    print("🔍 Prediction:", label)
    if probas is not None:
        print(f"📊 Confidence (HAM, SPAM): {probas.round(2)}")

# Example usage
while True:
    text = input("\n📝 Enter a message (or type 'exit'): ")
    if text.lower() == 'exit':
        break
    predict_message(text)
