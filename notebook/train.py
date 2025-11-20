# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier





df = pd.read_csv('credit_risk_dataset.csv')
strings = list(df.dtypes[df.dtypes == 'object'].index)
strings

for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')


df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'y': 1, 'n': 0})

# fill the missing numerical features with 0
df.person_emp_length = df.person_emp_length.fillna(0)
df.loan_int_rate = df.loan_int_rate.fillna(0)





# Look at numerical and categorical fetures
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


X_test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(X_test_dict)

# # Training Logistic Regression

# Training logistic regression with Scikit-Learn

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
y_pred

auc  = roc_auc_score(y_val, y_pred)

print(f'The AUC of this model on the validation dataset is {round(auc, 4)}')

for C in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:


    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]

    auc  = roc_auc_score(y_val, y_pred)

    print('%4s -> %.3f' % (C, auc))


# 1 seem to be the best value for C

# # Training Decision Trree

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)

# # Deision Tree Parameter Tuning

depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, None]

for depth in depths: 
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    print('%4s -> %.6f' % (depth, auc))

# 4 seems to be the best value for max_depth

scores = []

for depth in [4, 5, 6]:
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        scores.append((depth, s, auc))

columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)

sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")

# 200 seems to be the best value for min_samples_leaf

# %% [markdown]
# # Final decision tree model parameters

# %%
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc

# %%
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))


# %%

scores = []
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    scores.append((n, auc))

# %%
df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
plt.plot(df_scores.n_estimators, df_scores.auc)

# %% [markdown]
# Best value for n_estimator is 39

# %%
scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))

# %%
columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend()

# %% [markdown]
# Best value of max_depth is 15

# %%
max_depth = 15

scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))

# %%
columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

# %%
colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()

# %% [markdown]
# Best value for min_sample_leaf is 1

# %% [markdown]
# # Final Random Forest Model Parameters

# %%
rf = RandomForestClassifier(n_estimators=39,
                            max_depth=max_depth,
                            min_samples_leaf=1,
                            random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc

# %% [markdown]
# Training XGBoost

# %%
import xgboost as xgb

# %%
features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

# %%
xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)

# %%
y_pred = model.predict(dval)


# %%
roc_auc_score(y_val, y_pred)


# %% [markdown]
# XGBoost Parameter Tuning

# %% [markdown]
# # Selecting the best model

# %%
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc

# %%
rf = RandomForestClassifier(n_estimators=39,
                            max_depth=max_depth,
                            min_samples_leaf=1,
                            random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
auc

# %%
features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)

y_pred = model.predict(dval)

roc_auc_score(y_val, y_pred)



