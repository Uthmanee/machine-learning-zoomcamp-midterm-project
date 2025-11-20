# Loan Approval Prediction
This project develops a predictive model to determine the likelihood of loan application approval based on various applicant features, including age, income, housing status, loan purpose, and other relevant factors.

# Overview
The project involves data cleaning, exploratory data analysis (EDA), feature importance evaluation, model selection and evaluation.

# Dataset
In this project, Loan-Approval-Prediction-Dataset is used. Download it from [here](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

# Result
Best Model: Random Forest
AUC-ROC: 90%

# Installation
**1.If you do not have pipenv installed already, you can install it by with the command**
```
    pip install pipenv
```
**2.Clone this repository:**
```
git clone https://github.com/Uthmanee/machine-learning-zoomcamp-midterm-project.git
cd script
```
**3.Install all dependencies/packages mentioned in the **Pipfile** within the new virtual environment being created (as pipenv will prioritize Pipfile, requirements.txt has not been used)
```
pipenv install
```
**4.Build the Docker Image:**

Use the following command to build the Docker image
```
docker build -t loan-default .
```
**5.Run the Docker Container**

Start the container using
```
docker run -p 9696:9696 loan-default
```
