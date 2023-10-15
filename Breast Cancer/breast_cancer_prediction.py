import pandas as pd
import joblib

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

def predict_diagnosis(features):
    # Make a prediction
    prediction = model.predict([features])

    return "Malignant" if prediction[0] == 1 else "Benign"

if __name__ == "__main__":
    # Interactive input for feature values
    mean_radius = float(input("Enter mean_radius: "))
    mean_texture = float(input("Enter mean_texture: "))
    mean_perimeter = float(input("Enter mean_perimeter: "))
    mean_area = float(input("Enter mean_area: "))
    mean_smoothness = float(input("Enter mean_smoothness: "))

    # Create a feature list
    features = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]

    # Make a prediction
    result = predict_diagnosis(features)
    print(f"Predicted Diagnosis: {result}")
