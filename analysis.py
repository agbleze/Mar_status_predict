#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

df_g7sec1 = pd.read_spss(path='g7sec1.sav')

#%%

health = pd.read_spss(path='g7sec3b_Health_Insurance.sav')

#%%

g7sec7 = pd.read_spss(path='g7sec7.sav')

#%%

g7sec7j = pd.read_spss(path='g7sec7j.sav')

#%%

g7sec2 = pd.read_spss(path='g7sec2.sav')

#g7sec7j_sel = g7sec7j[[]]

#%%

df_g7sec1.columns

#%%
health.columns

#%%

g7sec7.columns

#%%

g7sec7j.columns


#%%

df_g7sec1['s1q6'].value_counts()


#%% 
df_g7sec1['s1q9'].count()

#%%

## used 'phid' as individuals in household

# select columns of interest for each dataset
# join data together on phid


#%%

df_sel = df_g7sec1[['phid', 'hid', 'clust', 'nh', 'pid', 's1q2', 's1q5y', 
                     's1q14', 's1q18', 's1q22', 's1q10', 's1q6'
                ]]

#%%

df_sel_sec2 = g7sec2[['phid', 'clust', 'nh', 'pid','s2aq1','s2aq1b']]


#%%
## s1q22 Number of months away from houshold
## s1q22 father live in household  <<  14 Does (NAME’S) father live in this household?
## s1q1b educational level <<  What is the highest level of education (NAME) has attained?


## s1q2 sex of individual
## s1q18  Mother live in household  <<  18 Does (NAME’S) mother live in this household?

## s1q5y Age in years

## s1q10 What is (NAME’S) religious denomination?





# %%

data = df_sel.merge(right=g7sec7j, on='phid', how='outer')

#%%

data = data.merge(right=df_sel_sec2, 
                  how='outer',
                  on='phid'
               )




# %%
data = data.rename(columns={'s7jq1': 'weight',
                     's7jq0': 'measured_or_not',
                     's7jq2': 'measure_mode',
                     's7jq3': 'Height',
                     'loc2':'urbrur',
                     's1q22':'months_away_from_hse',
                     's1q14':'father_in_hse',
                     's2aq1b':'highest_edu',
                     #'s1q2':'sex',
                     's1q18': 'mother_in_hse',
                     's1q5y': 'age_years',
                     's1q10': 'religion',
                     's2aq1': 'attend_school',
                     's1q6': 'marital_status'
                     }
            ).copy()

#%%

data.info()

#%%

data.describe()

#%%

data[['sex']]

#%%

data_mar_only = data.dropna(subset='marital_status')

#%%

data_mar_only[['weight', 'Height', 'age_yrs']].info()


#%%

from sklearn.preprocessing import LabelEncoder

#%%

label_encoder = LabelEncoder()

#%%

data_mar_only['marital_status_encode'] = label_encoder.fit_transform(data_mar_only['marital_status'])


#%%

from sklearn.ensemble import HistGradientBoostingClassifier,BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,classification_report,
                             roc_auc_score,precision_score,
                             recall_score,roc_curve,
                             balanced_accuracy_score
                             )
#from sklearn.

#%%

X = data_mar_only[['weight', 'Height', 'age_yrs']]

y = data_mar_only[['marital_status_encode']]

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2023, 
                                                    stratify=y['marital_status_encode'],
                                                    test_size=.3 #shuffle=True
                                                    )

#%%

histgb = HistGradientBoostingClassifier(class_weight='balanced')


#%%
histgb.fit(X_train, y_train)

#%%

accuracy_score(y_true=y_train, y_pred=histgb.predict(X_train))

#%% test accuracy

accuracy_score(y_true=y_test, y_pred=histgb.predict(X_test))

#%%

from tpot import TPOTClassifier

#%% initialize tpot with default parameters

tpot_classifier = TPOTClassifier(verbosity=3, 
                                 warm_start=True, 
                                 max_time_mins=60,
                                 use_dask=True,
                                 n_jobs=-1
                                 )

# %% train the classifier

tpot_classifier.fit(X_train, y_train)

#%%

tpot_classifier.pareto_front_fitted_pipelines_

#%%

tpot_classifier.evaluated_individuals_

#%%

tpot_classifier.fitted_pipeline_

#%%

accuracy_score(y_true=y_train, y_pred=tpot_classifier.predict(X_train))

#%%
accuracy_score(y_true=y_test, y_pred=tpot_classifier.predict(X_test))

# %% optimization techniques 










