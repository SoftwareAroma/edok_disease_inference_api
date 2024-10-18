import pickle

def predict_top_diseases(symptoms):
    # Load the model and vectorizer
    with open('models/disease_classifier.pkl', 'rb') as model_file:
        loaded_clf = pickle.load(model_file)

    with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    symptoms_tfidf = loaded_vectorizer.transform([' '.join(symptoms)])
    
    # Get the probabilities of each class (disease)
    probabilities = loaded_clf.predict_proba(symptoms_tfidf)
    
    # Get the class labels
    classes = loaded_clf.classes_
    
    # Create a dictionary of diseases and their probabilities
    disease_probabilities = {classes[i]: probabilities[0][i] for i in range(len(classes))}
    
    # Sort the diseases by their probabilities in descending order
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top 3 diseases and their likelihood percentages
    top_3_diseases = [(disease, prob * 100) for disease, prob in sorted_diseases[:3]]
    
    return top_3_diseases

# Example usage:
user_symptoms = ['facial pain', 'drastic weight loss', 'pain in stomach']
top_diseases = predict_top_diseases(user_symptoms)

# Print the results
for disease, likelihood_percentage in top_diseases:
    print(f'{disease}: {likelihood_percentage:.2f}%')