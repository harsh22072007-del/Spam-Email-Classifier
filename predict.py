import pickle

# Load saved model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Input message
message = input("Enter email message: ")

# Convert message to vector
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

if prediction[0] == "spam":
    print("This email is SPAM ❌")
else:
    print("This email is NOT SPAM ✅")
