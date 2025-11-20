import requests

url = 'http://localhost:9696/predict'
customer = {'person_age': 23,
 'person_income': 28000,
 'person_home_ownership': 'rent',
 'person_emp_length': 2.0,
 'loan_intent': 'education',
 'loan_grade': 'c',
 'loan_amnt': 12000,
 'loan_int_rate': 14.27,
 'loan_status': 1,
 'loan_percent_income': 0.43,
 'cb_person_cred_hist_length': 4}

response = requests.post(url, json=customer).json()
print(response)