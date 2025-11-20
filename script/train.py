# %%
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier



def load_data():

    df = pd.read_csv('credit_risk_dataset.csv')
    strings = list(df.dtypes[df.dtypes == 'object'].index)
    strings

    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ', '_')


    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'y': 1, 'n': 0})

    # fill the missing numerical features with 0
    df.person_emp_length = df.person_emp_length.fillna(0)
    df.loan_int_rate = df.loan_int_rate.fillna(0)

    return df


def train_model(df):
    categorical_features = list(df.dtypes[df.dtypes == 'object'].index)
    numerical_features = list(df.dtypes[df.dtypes != 'object'].index)

    # From the output below, numerical_fetures includes converted variable which is our target therefore the tagert variable has to be removed
    features = (categorical_features + numerical_features)
    features.remove('cb_person_default_on_file')


    X_train_dict = df[features].to_dict(orient='records')

    y_train = df.cb_person_default_on_file.values



    # # One Hot Encoding
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(X_train_dict)
    rf = RandomForestClassifier(n_estimators=39,
                                max_depth=15,
                                min_samples_leaf=1,
                                random_state=1)
    rf.fit(X_train, y_train)

    return dv, rf

def save_model(dv, model, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

df = load_data()
dv, model = train_model(df)
save_model(dv, model, 'model.bin')
