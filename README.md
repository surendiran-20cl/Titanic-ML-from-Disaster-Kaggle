

# Titanic - Machine Learning from Disaster

## Project Description

This project predicts passenger survival from the Titanic disaster using a machine learning model trained on the Titanic dataset from Kaggle. The workflow includes data preprocessing, feature engineering, model training, evaluation, and deployment of an interactive web application using Gradio on Hugging Face Spaces.

## Project Highlights

* Data cleaning and preprocessing
* Feature engineering based on domain knowledge
* Model development using Random Forest Classifier
* Model evaluation with accuracy and feature importance analysis
* Deployment of an interactive prediction app with Gradio
* Hosted publicly on Hugging Face Spaces

## Dataset

* Source: [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
* Files used:

  * train.csv
  * test.csv

## Tech Stack

| Tool                | Purpose                   |
| ------------------- | ------------------------- |
| Python              | Programming language      |
| pandas              | Data manipulation         |
| numpy               | Numerical operations      |
| scikit-learn        | Machine learning modeling |
| joblib              | Model serialization       |
| gradio              | Interactive web interface |
| Hugging Face Spaces | Deployment and hosting    |

## Model Features

| Feature    | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| Pclass     | Passenger class (1 = Upper, 2 = Middle, 3 = Lower)          |
| Sex        | Male or Female                                              |
| Age        | Passenger's age                                             |
| Fare       | Ticket fare                                                 |
| Embarked   | Port of embarkation (C, Q, S)                               |
| Title      | Extracted from passenger name (Mr, Miss, Mrs, Master, Rare) |
| FamilySize | Total number of family members aboard plus self             |
| IsAlone    | 1 if alone, 0 otherwise                                     |

## Application Demo

The web app allows users to input passenger details and receive predictions about survival probability.

Live demo: [Titanic Survival Prediction App](https://huggingface.co/spaces/surendirans/titanic-survival-app)
(Replace with your actual Hugging Face Space link)

## How to Use the App

1. Enter the passenger details such as class, gender, age, fare, embarkation port, title, family size, and whether they are traveling alone.
2. Click Submit.
3. The app will return whether the passenger is predicted to survive along with the prediction probability.

## Running Locally

### Clone the repository:

```bash
git clone https://github.com/YourUsername/titanic-ml-app.git
cd titanic-ml-app
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Gradio app:

```bash
python app.py
```

The application will launch locally at:

```
http://localhost:7860/
```

## Folder Structure

```
titanic-ml-app/
│
├── app.py                  # Gradio web app
├── titanic_model.pkl        # Serialized ML model
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore file
```

## Model Performance

* Model: Random Forest Classifier
* Accuracy: Approximately 80 percent
* Evaluation: Basic cross-validation and validation set accuracy
* Features selected based on exploratory data analysis and feature engineering

## Deployment

* The app is deployed on Hugging Face Spaces using the Gradio SDK.
* It is accessible via a web interface without requiring local setup.

## Future Improvements

* Implement more advanced machine learning models such as XGBoost or LightGBM
* Perform hyperparameter tuning
* Deploy as an API endpoint for backend integration
* Package the solution with Docker for cloud-native deployments

## License

This project is open-source and available for educational and learning purposes.

## Acknowledgements

* Kaggle Titanic Dataset
* Hugging Face for providing Spaces for hosting
* Gradio for the interactive machine learning interface
* scikit-learn for machine learning modeling

## Contact

* GitHub: [Surendiran GitHub](https://github.com/surendiran-20cl)
* LinkedIn: [Surendiran LinkedIn](https://linkedin.com/in/surendiran-shanmugasundaram)
