from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_model():
    vectorizer = TfidfVectorizer(stop_words='english')
    model = MultinomialNB()
    return vectorizer, model
