
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

# Handle any leftover nulls
df.dropna(subset=['cleaned_text'], inplace=True)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr = LogisticRegression(class_weight='balanced')
rf = RandomForestClassifier(class_weight='balanced')

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluate
lr_acc = accuracy_score(y_test, lr.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"[🔍] Logistic Regression Accuracy: {lr_acc}")
print(f"[🔍] Random Forest Accuracy: {rf_acc}")

# Choose best model
best_model = rf if rf_acc > lr_acc else lr
print(f"[🏆] {'Random Forest' if rf_acc > lr_acc else 'Logistic Regression'} selected.")

# Save model and vectorizer
joblib.dump(best_model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
print("[✅] Model and vectorizer saved.")