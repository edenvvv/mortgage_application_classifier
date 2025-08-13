import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Expanded sample data including "Other"
texts = [
    "I want to check mortgage rates",
    "How can I update my payment information?",
    "Where do I submit the documents?",
    "What is the status of my mortgage application?",
    "Tell me more about home loans",
    "What’s the weather like today?"
] # the last one is for "Other" category example

labels = [
    "Inquiry",
    "Payment",
    "Document",
    "Status",
    "Inquiry",
    "Other"
] # labels are category in the same order of the text list

# Load english spacy model
nlp = spacy.load("en_core_web_sm")

# Custom tokenizer using spaCy lemmatizer
def spacy_tokenizer(text):
    """
    Tokenised by Spacey,
    token.lemma_ = base or dictionary form. For example, the lemma of “running” is “run”, and the lemma of “better” is “good”.
    token.pos_ = part of speech, like noun, verb, adjective, adverb, etc.
    token.tag_ = part of speech and part of speech tag
    token.is_punct = True if the token is a punctuation mark
    token.is_stop = True if the token is a stop word
    """
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]

# Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

"""
Create pipeline: 
    1. TF-IDF Vectorizer: converts text to numeric vectors, using our custom tokenizer.
    2. Logistic Regression classifier: learns to classify text based on those vectors.
"""
model = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=spacy_tokenizer)),
    ('clf', LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

"""
### Evaluate on test data
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred)) # shows accuracy, precision, recall, and F1-score, giving insight into how well the model is performing
"""

# ----------- Interactive user input -------------
while True:
    new_sentence = input("\nEnter a sentence to classify (or 'exit' to quit): ").strip()
    if new_sentence.lower() == 'exit':
        print("Goodbye!")
        break
    if not new_sentence:
        print("Please enter some text.")
        continue

    # Predict category and probabilities
    predicted_category = model.predict([new_sentence])[0] # predicts the category label for the sentence
    probabilities = model.predict_proba([new_sentence])[0] # returns probabilities for all categories (how confident it is for each)
    categories = model.classes_ # stores all possible labels

    # Finds the probability (confidence) for the predicted category
    predicted_prob = probabilities[list(categories).index(predicted_category)]

    # Threshold for "not related" confidence, e.g. 0.25 (25%)
    threshold = 0.25

    print(f"\nInput: {new_sentence}")
    if predicted_category == "Other" or predicted_prob < threshold:
        print("This sentence does not seem related to mortgages.")
    else:
        print(f"Predicted category: {predicted_category} (confidence: {predicted_prob:.2f})")

    print("All category probabilities:")
    for cat, prob in zip(categories, probabilities):
        print(f"  {cat}: {prob:.2f}")
