# Diabetes Prediction Using Machine Learning

## Project Overview
This project is a Machine Learning-based approach to predict the likelihood of diabetes in patients. It uses the **Pima Indians Diabetes Database**, which consists of various medical predictor variables and a binary outcome indicating whether the patient has diabetes or not. The key focus of this project is to implement **K-Nearest Neighbors (KNN)** algorithm to build a model that predicts whether a patient is likely to have diabetes based on medical data.

## Dataset
The dataset used in this project is the **Pima Indians Diabetes Database**, which contains the following features:

- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skinfold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: A function that quantifies the likelihood of diabetes based on family history
- `Age`: Age in years
- `Outcome`: 0 for non-diabetic, 1 for diabetic (target variable)

## Project Workflow

1. **Data Exploration and Preprocessing**:
   - Load and inspect the dataset.
   - Handle missing values and duplicate entries.
   - Visualize the distribution of classes and individual features.
   - Identify and visualize outliers using boxplots.
   - Scale the features using **StandardScaler** for better performance with the KNN algorithm.

2. **Data Visualization**:
   - Generate count plots for the target variable (`Outcome`).
   - Use pair plots to visualize feature relationships with respect to `Outcome`.
   - Plot histograms to observe the distribution of features.
   - Create a heatmap to visualize the correlation between features.

3. **Model Building**:
   - Split the data into training and testing sets using **train_test_split**.
   - Implement the **K-Nearest Neighbors (KNN)** classifier.
   - Train the model and evaluate its performance for different values of `k` (number of neighbors).
   - Visualize training and testing accuracies for varying `k` values.

4. **Model Evaluation**:
   - Evaluate the model performance using a **confusion matrix**.
   - Generate a **classification report** to assess the precision, recall, and F1-score of the model.

## Conclusion
This project demonstrates the use of the **K-Nearest Neighbors (KNN)** algorithm in predicting diabetes, utilizing a well-prepared dataset. Further improvements could involve testing other machine learning models and fine-tuning hyperparameters for better performance.
