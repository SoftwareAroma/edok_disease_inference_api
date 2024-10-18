import pickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json

app = FastAPI()

def predict_top_diseases(symptoms):
    try:
        # Load the model and vectorizer
        with open('models/disease_classifier.pkl', 'rb') as model_file:
            loaded_clf = pickle.load(model_file)

        with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model or vectorizer file not found.")
    except pickle.UnpicklingError:
        raise HTTPException(status_code=500, detail="Error loading the model or vectorizer.")
    
    # Vectorize the input symptoms
    symptoms_tfidf = loaded_vectorizer.transform([' '.join(symptoms)])
    
    # Get the probabilities of each class (disease)
    probabilities = loaded_clf.predict_proba(symptoms_tfidf)
    
    # Get the class labels
    classes = loaded_clf.classes_
    
    # Create a dictionary of diseases and their probabilities
    disease_probabilities = {
        classes[i]: probabilities[0][i] for i in range(len(classes))
    }
    
    # Sort the diseases by their probabilities in descending order
    sorted_diseases = sorted(
        disease_probabilities.items(), key=lambda x: x[1], reverse=True
    )
    
    # Get the top 3 diseases and their likelihood percentages
    top_3_diseases = [
        (disease, round(prob * 100 * 100, 2)) for disease, prob in sorted_diseases[:10]
    ]
    
    return top_3_diseases


@app.get('/')
def read_root():
    return JSONResponse({
        'message': 'Welcome to the Disease Prediction API'
    })

@app.get('/predict_diseases')
def predict_diseases(symptoms: str):
    # Validate input: Ensure symptoms are provided
    if not symptoms:
        raise HTTPException(status_code=400, detail="Please provide symptoms as a comma-separated string.")

    # Split the symptoms string into a list
    user_symptoms = [symptom.strip() for symptom in symptoms.split(',') if symptom.strip()]
    
    # Validate that at least one symptom is provided
    if not user_symptoms:
        raise HTTPException(status_code=400, detail="Please provide at least one valid symptom.")

    # Get the top diseases predicted by the model
    top_diseases = predict_top_diseases(user_symptoms)
    
    return {
        'top_diseases': top_diseases
    }

@app.get('/symptoms')
def get_symptoms():
    try:
        # Load the symptoms from the symptoms.json file
        with open('models/symptoms.json', 'r') as file:
            data = json.load(file)
        
        # Check if the 'symptoms' key exists
        if 'symptoms' not in data:
            raise HTTPException(status_code=500, detail="Symptoms data is incorrectly formatted.")
        
        # Return the unique symptoms list
        return {
            'symptoms': data['symptoms']
        }
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Symptoms file not found.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error reading the symptoms file.")
    
    
@app.get('/treatment')
def get_treatment(disease: str):
    # Load the treatment data from the JSON file
    try:
        with open('models/treatment.json', 'r') as file:
            treatments = json.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Treatment file not found.")
    
    # Find the treatment for the given disease
    for treatment in treatments:
        if treatment['Disease'].lower() == disease.lower():
            return {
                "Disease": treatment['Disease'],
                "Pharmacological Treatment": treatment['Pharmacological Treatment'],
                "Non-Pharmacological Treatment": treatment['Non-Pharmacological Treatment']
            }
    
    # If the disease is not found, return an error
    raise HTTPException(status_code=404, detail="Treatment for the specified disease not found.")
    
    
# Example usage:
# user_symptoms = 'facial pain,drastic weight loss,pain in stomach'
# top_diseases = predict_top_diseases(user_symptoms)

# start the server
# uvicorn app:app --reload