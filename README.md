# DataMedic AI

DataMedic AI is a general-purpose AI-powered application for analysing CSV datasets and supporting a complete machine-learning workflow.

The application allows users to upload a CSV dataset, inspect its quality, select a prediction target, automatically prepare the data, compare suitable machine-learning models, explain the selected model using SHAP, and generate a downloadable PDF report.

The system is designed to work with different datasets rather than being tied to one specific dataset.

---

## Features

### 1. Dataset Analysis

DataMedic AI analyses the uploaded dataset and provides information about:

- Number of rows and columns
- Missing values
- Duplicate rows
- Data types
- Potential outliers
- Highly correlated numerical features
- Target-class balance when a target is selected
- Overall data-quality score

The analysis is designed to help users understand the condition of their data before applying machine learning.

---

### 2. Automatic Data Preprocessing

The preprocessing pipeline is designed to work with different types of CSV datasets.

It can:

- Detect whether the task is classification or regression
- Handle missing values
- Convert numeric-like text columns when appropriate
- Remove constant columns
- Remove ID-like text columns
- Scale numerical features
- Encode categorical features using one-hot encoding
- Split the dataset into training and testing sets

---

### 3. Machine Learning

After the user selects a target column, DataMedic AI automatically determines the type of prediction problem.

#### Classification

Classification is used when the target represents categories.

The system compares:

- Random Forest
- Logistic Regression
- Decision Tree

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

#### Regression

Regression is used when the target represents a continuous numerical value.

The system compares:

- Random Forest
- Linear Regression
- Decision Tree

The models are evaluated using:

- MAE
- RMSE
- R²

The best-performing model is automatically selected according to the primary metric.

---

### 4. Explainable AI with SHAP

DataMedic AI uses SHAP (SHapley Additive exPlanations) to make machine-learning predictions easier to understand.

The SHAP module:

- Selects an appropriate explainer for the trained model
- Calculates SHAP values
- Calculates global feature importance
- Generates a SHAP summary plot
- Identifies features that have the strongest influence on the model

This provides an Explainable AI component instead of treating the machine-learning model as a black box.

---

### 5. Web Interface

The application provides a Flask-based web interface where users can:

1. Upload a CSV dataset
2. Review dataset information
3. Select a target column
4. Run machine-learning analysis
5. Compare model results
6. View SHAP explanations
7. Access the final results

---

### 6. PDF Reporting

DataMedic AI can generate a downloadable PDF report containing important analysis and machine-learning results.

The report is designed to make the results easier to save, review and share.

---

## Project Workflow

The general workflow is:

```text
CSV Dataset
     |
     v
Dataset Upload
     |
     v
Dataset Analysis
     |
     v
Target Column Selection
     |
     v
Data Preprocessing
     |
     v
Classification / Regression Detection
     |
     v
Model Training
     |
     v
Model Comparison
     |
     v
Best Model Selection
     |
     v
SHAP Explainability
     |
     v
Results & PDF Report