## Loan Approval Prediction
This project builds a predictive model that estimates the likelihood of a loan application being approved based on key applicant attributes such as age, income, housing status, loan purpose, and other relevant financial and demographic factors. By analyzing these features, the model provides a data-driven assessment to support more efficient and consistent lending decisions.

## Problem Statement
Financial institutions receive large volumes of loan applications and must evaluate each applicant’s creditworthiness accurately and consistently. Manual assessment can be time-consuming, subjective, and prone to inconsistencies. Therefore, there is a need for an automated system that can reliably predict whether a loan application is likely to be approved based on applicant characteristics.
The objective of this project is to develop a machine learning model that analyzes applicant data and predicts loan approval outcomes, helping streamline the decision-making process and improve operational efficiency

## Overview
The project involves data cleaning, exploratory data analysis (EDA), feature importance evaluation, model selection and evaluation. Multiple models were trained and tuned, and the best-performing model was selected based on evaluation metrics (AUC-ROC)

## Dataset
In this project, Loan-Approval-Prediction-Dataset is used. Download it from [here](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

## Result
Best Model: Random Forest
AUC-ROC: 90%

### Installation & Setup
**1. Clone this repository:**
```
git clone https://github.com/Uthmanee/machine-learning-zoomcamp-midterm-project.git
cd script
```
**2. Install Pipenv (if not already installed)**
```
    pip install pipenv
```
## Run locally (without docker)
**1. Install all dependencies/packages mentioned in the **Pipfile** within the new virtual environment being created (Recommended)**
```
pipenv install
```
or, if you prefer to install from requirements.txt:
```
pipenv install -r requirements.txt
```
**2. Run the flask server**
```
# Enter the virtual environment by running the command below in the root directory
pipenv shell

# Start the flask server by running
python predict.py
```
**3. Run the test script to make a prediction**
```
# Make a prediction
python predict_test.py
```

## Docker Usage
**1. Build the Docker Image:**

Use the following command to build the Docker image
```
docker build -t loan-default .
```
**2. Run the Docker Container**

Start the container using
```
docker run -p 9696:9696 loan-default
```
**3. Run the test script to make a prediction**
```
# Make a prediction
python predict_test.py
```