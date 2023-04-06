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

data = df_sel.merge(right=g7sec7j, on='phid', how='outer').merge(right=df_sel_sec2, 
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
                     's1q2':'sex',
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

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

#%%

X = data_mar_only[['weight', 'Height', 'age_yrs']]

y = data_mar_only['marital_status']

# %%
