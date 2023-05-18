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

df_sel = df_g7sec1[['phid', 'hid', 'clust', 'nh', 'pid', #'s1q2', 
                    's1q5y', 
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

# TODO: 
# SELECT all predictors for exploration
# Undertake data visualization for target and predictors
# undertake feature selection
# identify possibility of feature engineering



#%%
"""
# Introduction - Problem statement

DateRush Mate (hypothetical firm used to establish a business case for this data science project) aims to be one of the 
leading platform providers for dating services hence has a keen focus on continuous improvement of their products. 
One of such servces is aiding matches make better decisions on their potential partners before taking in-person dates and 
establishing strong bonds. While openness is encouraged as the cornerstone of building trust among matches, not every platform 
user provides the same amount of information. Helping bridge the gap and developing tools to provide such information has 
make a solid business case so much so that we have have segmented users into different tiers with Pro users being provided 
the services of filling in the blanks on information that a match has not provided and is not required to do 
so to start with. For this, intelligent tools are required for prediction.


In providing such services, the most requested information by our users is to know the marital status of client. Given that 
a platform user is not obliged to include their the marital status on the profile but have the option of asking their match 
in person, many users leave-out such information yet consider knowing that of a partner important in deciding whether an in-person 
date or serious dating should be considered. Driven by solving problems that are most important and whose solutions are most 
requested by users, predicting the marital status of users is designated as a high priority task for which a demonstration of 
how to develop a working solution is provided in this post. 


The understanding gained from the problem statement as well as our background as data scientists informs our judgement that 
a machine learning solution is required rather than beating information or confession out of our clients. Thus, the focus of 
this discussion is on demonstration how to develop a machine learning solution for the given problem. 

As can be deduced already, we have different stakeholders and collaborators working together at different stages to devise 
a solution. Identifying these stakeholders and partners is critical for the success of the project. In particular, this enables
defining clear cut-out deliverables and requirements upon which all parties can determine success status for the project. 
The stakeholders for this project are identified to be product owner and date counsellors whose services are 
requested by some users after matching and proceeding for serious relationship. The collaborators are the data scientist and 
Machine Learning Engineer. The most discernible difference between the stakeholders which also poses a challenge to resolve is 
the fact that they have different technical backgrounds. By this, the end-users of the product are non-technical audience 
who need the easiest way to use the solutions without any coding required while the Data Scientist will write codes for which he/she 
can explain the logic to the non-technical audience when the need arises.

In response to providing such medium for capturing the project as a snapshot for reference among all stakeholders, a problem 
design framework is used to capture the problem statement and project for that matter. For this the Situation Complication Question 
Answer (SCQA) is used to capture the problem statement as a snapshot. This allows for easy communication of the project to 
non-technical users. The SCQA is depicted below;


## Research Question 

(i) How do we identify a dataset with features that we can collect data for on our plantform for developing an accurate model?

(ii) How do we leverage available data to predict marital status of people?


## Research Objective

(i) To identify open source data with features for predicting marital status

(ii) To develop a machine learning model for predicting marital status

 

## Search for Data for modelling

Data science projects start with data collection and for most business cases, data is usually provided by the business 
instead of field data collection. Instances where data is not available requires usually an open source data as one of the 
approaches to developed a model. In such cases, where we will still need to increase the sample size of the data and improve 
the model in the future, focus is drawn to which of the features in the open source data can the business generate its own 
data for. Using dataset with features for which data cannot be generated internal will come with an extra challenge and cost.
Hence, even though several open-source data  may be available not all of them will be useful in case, there still need to 
be used to meet business needs in the future. For example, if a data has blood group or HIV/AIDS status as one of the feature 
and we can not generate data for that feature in our business in the future then we should not use that feature for modelling 
to meet business needs. Features in dataset for modelling should be one that will be available in the future when new data 
is available. This inform the search for dataset and features for developing the modelling.


The dataset used for developing the machine learning model is the Ghana Living Standards Survey 7 (2016/2017). 
The dataset can be downloaded from https://www2.statsghana.gov.gh/nada/index.php/catalog/97/study-description. 
The dataset contains data on marital status and other features which are preselected to be explored for developing the model.


## Understanding the dataset / data collection method

Before even starting the data exploration, it is critical that we familiarize ourselves with the metadata and 
questionnaires to understand the various variables and their background. The files from which the variables were 
extracted are mainly from g7sec1.sav, g7sec7j, and g7sec2. These files are found in the downloaded dataset are 
identified to the relevant ones for this analysis. A review of the variables and meaning from these files is provided
below. For each variable, the corresponding questions asked to collect the data is provided

• s1q2: sex of individual

• s1q18:  'mother_in_hse'  <<  18 Does (NAME’S) mother live in this household?

• s1q10: 'religion'  <<  What is (NAME’S) religious denomination?

• s7jq3: 'Height'  <<  Height (cm) of household member.

• s1q14:  'father_in_hse'   <<   Does (NAME’S) father live in this household?

• s1q22:  'months_away_from_hse'   <<  For how many months during the past 12 months has (NAME) been continuously away from this household?

• s2aq1b: 'highest_edu'   <<   What is the highest level of education (NAME) has attained?

• 's2aq1': 'attend_school'  <<  Has (NAME) ever attended school?

• 's1q6': 'marital_status'  <<   What is (NAME’S) present marital status?

• 's7jq1': 'weight'  <<   Weight (Kg) of household member

• phid: id of individual in household

• age_yrs

• sex



### Identifying and categorizing variables in the data

The data type used to store a variable determines the typs of exploratory analysis, visualization and categorical machine 
learning algorithm to use. Thus, the various variables are accessed based on that as follows:


#### Target / Outcome variable

• marital_status: Categorical

#### Predictor / feature variables

• mother_in_hse: Categorical

• religion: Categorical 

• Height: Continuous quantitative 

• father_in_hse: Categorical 

• highest_edu: Categorical 

• attend_school:  Categorical 

• weight:  Continuous Quantitative

• months_away_from_hse: Discrete Quantitative

• age_yrs: Discrete Quantitative

• sex: Categorical 

Exploratory data analysis is organized to reflect the data types of the various variables as follows.


# Exploratory analysis

Exploratory data analysis is undertaken to investigate some of the assumptions that some machine learning algorithms 
require. Among others, the type of underlying distribution that a variable is from, the relationsip between the 
various variables and possible data transformation are explored at this stage. The insights gained from such analysis 
is important for undertaking feature selection.


### Data analysis 

Given the various features preselected to be explored for modelling are in different files, 
there is the need to combine data from the various files. For this, a common variable in the files is 
required and "phid" variable is used for that purpose. Afterwards the variables are rename into something
more understandable.

## Data Visualization: Relationship between variables



"""

#%% import all modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier,BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,classification_report,
                             roc_auc_score,precision_score,
                             recall_score,roc_curve,
                             balanced_accuracy_score
                             )


#%%

#g7sec7 = pd.read_spss(path='g7sec7.sav')

df_g7sec1 = pd.read_spss(path='g7sec1.sav')
g7sec7j = pd.read_spss(path='g7sec7j.sav')  ###
g7sec2 = pd.read_spss(path='g7sec2.sav')

df_sel = df_g7sec1[['phid', 'hid', 'clust', 'nh', 'pid', #'s1q2', 
                    's1q5y', 
                     's1q14', 's1q18', 's1q22', 's1q10', 's1q6'
                ]]

df_sel_sec2 = g7sec2[['phid', 'clust', 'nh', 'pid','s2aq1','s2aq1b']]

data = df_sel.merge(right=g7sec7j, on='phid', how='outer')


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

""" 
One of the factors that influences the kind of analysis that a variable can be subjected to is the 
data type. Hence the data type of the various variables are determined as follows

"""

data.info()

# From the above it is deduced that missing data is present in some of variables and their data types are identified.

#%%
"""

###   Descriptive statistics 

Most descriptive statistics such mean, minimum, maximum among others, highlight the range and 
distribution of variables that are quantitative. Hence quantitative variables are selected for this 
type of analysis as follows.

"""
#%%
data[['age_years', 'Height', 'weight']].describe()

"""
From the analysis, the mean gae is about 25 years. The high difference between the 75% percentile
age (37) and maximum age (99) suggests that outliers is likely present and has to be treated. The 
same deductions are made for height and weight. A number of ways can be used to handle outliers 
when identified and this generally involves choosing an algorithm that is robust to it or imputing 
it. 
"""

#### Missing data




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










