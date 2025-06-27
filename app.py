import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('titanic_model.pkl')

def predict(Pclass, Sex, Age, Fare, Embarked, Title, FamilySize, IsAlone):
    sex = 1 if Sex == 'male' else 0
    embarked = {'C':0, 'Q':1, 'S':2}[Embarked]
    title = {'Mr':2, 'Miss':1, 'Mrs':3, 'Master':0, 'Rare':4}[Title]
    is_alone = int(IsAlone)

    data = pd.DataFrame([{
        'Pclass': int(Pclass),
        'Sex': sex,
        'Age': float(Age),
        'Fare': float(Fare),
        'Embarked': embarked,
        'Title': title,
        'FamilySize': int(FamilySize),
        'IsAlone': is_alone
    }])

    prediction = model.predict(data)[0]
    prediction_proba = model.predict_proba(data)

    result = 'Survived' if prediction == 1 else 'Not Survived'
    return result, f"Probability (Not Survived): {round(prediction_proba[0][0]*100, 2)}%, (Survived): {round(prediction_proba[0][1]*100, 2)}%"
    

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Passenger Class (1=Upper, 2=Middle, 3=Lower)"),
        gr.Radio(['male', 'female'], label="Sex"),
        gr.Slider(0, 100, value=30, label="Age"),
        gr.Slider(0, 500, value=50, label="Fare"),
        gr.Radio(['C', 'Q', 'S'], label="Port of Embarkation"),
        gr.Radio(['Mr', 'Miss', 'Mrs', 'Master', 'Rare'], label="Title"),
        gr.Slider(1, 10, value=1, label="Family Size"),
        gr.Radio([0, 1], label="Is Alone (1=Yes, 0=No)")
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Prediction Probability")
    ],
    title="🚢 Titanic Survival Prediction",
    description="Enter the passenger details to predict whether they would have survived the Titanic disaster.",
)

iface.launch()
