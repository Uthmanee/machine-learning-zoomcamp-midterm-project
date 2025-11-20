import numpy as np
import pickle

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

X_val = {'person_age': 23,
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

X_val_transformed = dv.transform([X_val])
print(model.predict_proba(X_val_transformed)[:, 1])