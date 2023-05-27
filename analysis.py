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

Before proceeding with it, one cannot resist addressing a frequently asked question concerning 
when it should be undertaken. Yes, it is always the first stages but lets talk about the its focus.
Very often the question is asked;

Should the dataset be splitted before undertaking the exploratory analysis?

Should exploratory analysis be undertaken on all the dataset or only the training dataset?

Well, these two questions are driving at the same thing from the same angle.

The answers are Yes and No with solid arguemnt for both so that it all truns down to be 
where you see things from base on your situation. I will argue for each point to present 
the views based on which you decide.


Yes, the dataset should be splitted before exploratory analysis!
Exploratory analysis should be based on only training data!






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

from plotnine import (ggplot, ggtitle, aes, geom_col, theme_dark, 
                      theme_light, scale_x_discrete,
                      position_dodge, geom_text, element_text, theme,
                      geom_histogram, geom_bar, xlab, ylab, scale_y_log10, scale_x_log10,
                      geom_point, geom_smooth, geom_boxplot, coord_flip
                    )

from scipy.stats import iqr, shapiro
import scipy.stats as stats


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
                     's7jq3': 'height',
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
data_selected = data[['father_in_hse', 'mother_in_hse', 'months_away_from_hse',
                         'marital_status', 'sex', 'age_yrs', 'weight', 'height', 
                         'attend_school', 'highest_edu'
                         ]]

data_selected.info()

""" 
From the above the data types of various variables are identified and it is deduced 
that missing data is present in some of variables. However, given that the whole analysis
is based on the number of data points for target variable, only instances where 
data is available for marital status will be considered and afterwards the presence of
missing data will checked.

"""

#%%

marital_status_df = data_selected.dropna(subset='marital_status')

marital_status_df.info()


""" 
A key insight gained from the data after reducing it to only observations where marital 
status is recorded is that most of the variables have missing data 
hence have to be handle as part of the data preparation process.

The decision on how to handle missing data is equally influence by the percentage 
of data that is missing given that imputing a high percentage of missing data introduces 
a high amount of "artificial data" that can change the actual underlying distribution of 
the phenomenon being studied. The percentage of data missing for each variable is 
computed as follows.

"""
#%%

# Total missing data as a percentage of all data points is estimated as follows
def get_missing_data_percent(data: pd.DataFrame, variable: str):
    total_missing = data[variable].isnull().sum()
    total_data = data.shape[0]
    percent_missing = (total_missing / total_data) * 100
    print(f'Percentage of data missing in {variable}: {round(percent_missing, 5)}%')

#%% # implement function for percentage of missing data per variable
for variable_name in marital_status_df.columns:
    get_missing_data_percent(data=marital_status_df, variable=variable_name)
    
 
#%% 
"""  
From the analysis of missing data, highest education level has as much as 25.47% 
missing data and weight and height variable has 20% missing data.  Arguably, this will 
be a lot of data to impute and is likely to influence the underlying distribution with so many 
"artificial data point" to be introduced when imputed. Other variables such as father_in_hse, 
mother_in_hse, sex, age among others have less than 2% of data points missing hence can be imputed 
without altering the variable distribution much. The insights gained from the proportion of 
missing data available in the various variables will be used to determine how to handle or 
preprocess missing data.

 
""" 
 
    

#%%
"""

###   Descriptive statistics 

Most descriptive statistics such mean, minimum, maximum among others, highlight the range and 
distribution of variables that are quantitative. Hence quantitative variables are selected for this 
type of analysis as follows.

"""
#%%
marital_status_df[['age_yrs', 'height', 'weight']].describe()

#%%
"""
From the analysis, the mean age is about 34 years. The high difference between the 75% percentile
age (46) and maximum age (99) suggests that outliers is likely present and has to be treated. The 
same deductions are made for height and weight. A number of ways can be used to handle outliers 
when identified and this generally involves choosing an algorithm that is robust to it or imputing 
it. 
"""

### Visualizing the distribution of numeric variables

"""
Some algorithms assume that predictors are normally distributed hence testing such the validity 
of such an assumption in our dataset in undertaken. Visualization and statistical approaches are 
used for. First, histogramm is used a visualization technique to determine if the distribution 
is normal.

"""
#%%

def plot_histogram(data: pd.DataFrame, variable_to_plot: str, 
                   title: str = None, bins_method: str = 'freedman-Diaconis'
                   ):
    data = data.dropna(subset=variable_to_plot).copy()
    
    # by default Freedman–Diaconis rule is computed to determine 
    #optimal number of bins for histogram
    
    if bins_method == 'freedman-Diaconis':
        h = (2 * iqr(np.array(data[variable_to_plot].dropna().values))) / (len(data[variable_to_plot])
              **(1/3)
            )

        nbins = (data[variable_to_plot].max() - data[variable_to_plot].min()) / h
        nbins = round(nbins, 1)
    else:
        nbins = 5
        
    if title is None:
        title = f"Distribution of {variable_to_plot}"
    histogram = (ggplot(data, aes(x=variable_to_plot))
                + geom_histogram(bins=nbins)
                + ggtitle(title) 
            )
    return print(histogram)


#%%

# plot of quantitative predictor variables

numeric_predictors = ['age_yrs', 'height', 'weight']


for var in numeric_predictors:
    plot_histogram(data=marital_status_df, variable_to_plot=var)


"""
From the histogram, some level of skewness is present in all variables with age and weight 
appearing to be slightly right skewed while hieght is left skewed. The skewness is not very 
pronounce to say for certainty the distribution is not normally distributed as such conclusions
could more subjective than objective. In such instances statistical test is needed.
It is possible to use boxcox transformation to transform them into normal variables 
should that be needed for a linear algorithm.

"""

### Statistical analysis to determine normality of numeric predictor distributions
"""
In addition to histogram visualization, Shapiro Wilk test is a statistical analysis that 
can used to test the normality of numeric variable distribution. It tests the null hypothesis 
that the variable has a normal distribution. The Shapiro test is used as a more objective 
approach to determining normality hence supplemnts histogram visualization. Shapiro test 
is implemented as follows:

"""

#%%
# shapiro test
def compute_shapiro_normality_test(data: pd.DataFrame, variable_name: str,
                                   sig_level: int = 0.05
                                   ):
    shapiro_result = shapiro(data[variable_name])
    p_value = shapiro_result[1]
    sig_level_statement = f"at {sig_level * 100}% significance level"
    if p_value <= sig_level:
        shapiro_conclusion = "reject Null hypothesis of normal distribution"
    else:
        shapiro_conclusion = "fail to reject Null hypothesis of normal distribution"
        
    shapiro_interprete = f"With a p-value of {p_value} the shapiro test suggests to: {shapiro_conclusion} {sig_level_statement}"
    print(f"Shapiro Wilk test result of {variable_name}")
    print(shapiro_interprete)
    
    
for var in numeric_predictors:
    compute_shapiro_normality_test(data=marital_status_df, variable_name=var)
        
#%%    
    
### Visualizing relationship between numberic variables and target variable
"""
Bar plot can be used to visualize how numeric predictors such as age varies among the 
various categories of marriage status (target variable) on the average. This technique 
enables gaining an insight into the relevance of a predictor to predicting marriage status.
First, the data is prepared for visualization.
"""

#%%

avg_predicors_per_marital_status = (marital_status_df.groupby(by='marital_status')
                                    [numeric_predictors].agg('mean').reset_index()
                                    )


#%%
#### Now, the function for plotting it is defined as follows:

# barplot 
def barplot(data_to_plot: pd.DataFrame, 
            variable_to_plot: str, y_colname: str = None
            ):
    title = f'Average {y_colname} per {variable_to_plot}'
    ylabel = f'Average {y_colname}'
    if y_colname is None:
        bar_graph = (ggplot(data=data_to_plot, mapping=aes(x=variable_to_plot, 
                                                fill=variable_to_plot
                                            )
                                )
                                + geom_bar()  
                                + ggtitle(title) + xlab(variable_to_plot)
                                + ylab(ylabel)
                        )

        return print(bar_graph)
    else:
        bar_graph = (ggplot(data=data_to_plot, 
                            mapping=aes(x=variable_to_plot, 
                                        y=y_colname,
                                        fill=variable_to_plot
                                        )
                                )
                                + geom_col()
                                + ggtitle(title) + xlab(variable_to_plot)
                                + ylab(ylabel)
                        )

        return print(bar_graph)

#%%
for var in numeric_predictors:
    barplot(data_to_plot=avg_predicors_per_marital_status,
            variable_to_plot='marital_status', y_colname=var
            )




    
    
 
 
    
 #%% test of homogeneity of variance
def test_homogeneity(data: pd.DataFrame, target_var: str, predictor_var: str):
    infostat_test = infostat()
    sig_level = f'at 5% significance level'
    infostat_test.bartlett(df=data, res_var=target_var, xfac_var=predictor_var)
    bartlett_summary = infostat_test.bartlett_summary
    bartlett_pval = bartlett_summary[bartlett_summary['Parameter'] == 'p value']['Value'].item()
    
    if bartlett_pval <= 0.05:
        bart_res = 'reject Null hypothesis of equal variance'
    else:
        bart_res = 'fail to reject Null hypothesis of equal variance'
        
    bartlett_interprete = f'With a p-value of {bartlett_pval} the bartlett test suggests to: {bart_res} {sig_level}'
    
    infostat_test.levene(df=data, res_var=target_var, xfac_var=predictor_var)
    levene_summary = infostat_test.levene_summary
    levene_pval = levene_summary[levene_summary['Parameter'] == 'p value']['Value'].item()
    
    if levene_pval <= 0.05:
        levene_res = 'reject Null hypothesis of equal variance'
    else:
        levene_res = 'fail to reject Null hypothesis of equal variance'
        
    levene_interprete = f'With a p-value of {levene_pval}, the Levene test suggests to: {levene_res} {sig_level} '
    
    # results are printed and not return but in case of production environment they will be return
    print(f'Barlett test results of {predictor_var}')
    print(f'{bartlett_summary} \n')
    
    print(f'Levene test results of {predictor_var}')
    print(f'{levene_summary} \n')
    
    print(f'{bartlett_interprete} \n')
    print(f'{levene_interprete} \n')
    
    
       
    
    
    
    




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










