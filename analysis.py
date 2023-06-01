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

from scipy.stats import iqr, shapiro, chisquare
import scipy.stats as stats
import plotly.express as px
from bioinfokit.analys import stat as infostat

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
"""
###   Visualizing outliers

#### Boxplot to visualize outliers
One of the exploratory analysis that can be on numeric variables is to visualize, detect and even 
treat outliers. Infact, some algorithms are impacted by the presence of outliers hence analyzed 
to make an inform decision on which class of algorithm to choose from. 

To visualize an outlier, boxplot is used. 

"""    

#%%
# function to create boxplot
def make_boxplot(data: pd.DataFrame, variable_name: str):
    """This function accepts a data and variable name and returns a boxplot

    Args:
        data (pd.DataFrame): Data to visualize
        variable_name (str): variable to visualize with boxplot
    """
    fig = px.box(data_frame=data, y = variable_name,
                 template='plotly_dark', 
                 title = f'Boxplot to visualize outliers in {variable_name}',
                 height=700
                 )
    fig.show()


#%% plot outliers
for var in numeric_predictors:
    make_boxplot(data=marital_status_df, variable_name=var)    
    

"""
While boxplot have been used to as a graphical method to identify and 
visualize outliers so far, a number of statistical techniques exist for 
detecting outliers. 
This includes using the standard deviation method whereby data points 
that are more than 3 standard deviations  are considered outliers. 
Interquantile range is equally used as a technique for identifying 
outliers. In this case, a value is regarded as an outlier when it is 
greater than 1.5 times the interquartile range above the third quartile 
or below the first quartile. For this, upper limits and lower limits 
need to be define for capping outliers. 

A number of strategies exist to treat outliers and this includes trimming, 
capping, imputing with mean or other forms of descriptive statistics and 
various forms of transformation such as logarithm and square root among others. 


The interquartile range method is employed here to identify outliers and 
winsorization is used to treat outliers. In this case, outliers are set to 
the upper limit and lower limit defined after estimating the interquantile range times 1.5.

From the boxplots, height and weight have outliers above the upper limit and 
below the lower limit while age has outliers only above the upper limit and this 
decution needs to statistically established beyond visualization. To do this,
a class is implemented to accept data and variable to be analyzed. The class 
will have behaviours such as functions for identifying and getting outlier samples and 
imputing outliers. Certain attributes such as upper 
limit and lower limit used among others would also be worth accessing for perusal 
by the user. This is implemented below.
"""

#%%
class OutlierImputer(object):
  def __init__(self, data: pd.DataFrame, colname: str):
    self.data = data
    self.colname = colname
    self.first_quantile = self.data[self.colname].quantile(q=0.25)
    self.third_quantile = self.data[self.colname].quantile(q=0.75)
    self.inter_quantile_rng = 1.5*(self.third_quantile-self.first_quantile)
    self.upper_limit = self.inter_quantile_rng + self.third_quantile
    self.lower_limit = self.first_quantile - self.inter_quantile_rng

  @property
  def get_outlier_samples(self):
    outlier_samples = (self.data[(self.data[self.colname] > self.upper_limit) | 
                               (self.data[self.colname] < self.lower_limit)]
                              [[self.colname]]
                      )
    return outlier_samples



  def impute_outlier(self):
    self.outlier_data = self.data.copy()
    self.outlier_data[f'{self.colname}_outlier_imputed'] = (
                    np.where(self.outlier_data[self.colname] > self.upper_limit, 
                                                   self.upper_limit, 
                             np.where(self.outlier_data[self.colname] < self.lower_limit, 
                                                   self.lower_limit, 
                             self.outlier_data[self.colname]
                            )
                            )
                  )
    
    return self.outlier_data
    

#%%
for var in numeric_predictors:
  outlier_imputer = OutlierImputer(data=marital_status_df, colname=var)
  print(f'First quantile of {var}: {outlier_imputer.first_quantile}')
  print(f'Third quantile of {var}: {outlier_imputer.third_quantile}')
  print(f'Lower limit of {var}: {outlier_imputer.lower_limit}')
  print(f'Upper limit of {var}: {outlier_imputer.upper_limit}')
  print(f'Number of outliers in {var}: {len(outlier_imputer.get_outlier_samples)}')
  marital_status_df = outlier_imputer.impute_outlier()
  make_boxplot(data=marital_status_df, 
               variable_name=f'{var}_outlier_imputed'
               )


#%%
"""
From the results of implementing the Outlier class, it is evident that outliers are no more 
present of identifying and treating them.
"""
    
        
#%%    
## Feature selection for numeric variables    
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
                                + theme(axis_text_x=element_text(rotation=45, hjust=1))
                        )

        return print(bar_graph)

#%%
for var in numeric_predictors:
    barplot(data_to_plot=avg_predicors_per_marital_status,
            variable_to_plot='marital_status', y_colname=var
            )


"""
Generally, marked difference in the average predictor value between 
categories of the target variable suggests that the predictor is likely to 
be a good discriminator between target variable categories hence a strong 
relationship exist making the predictor relevant for the modelling experiment.
The contrary where the difference is not significant, then the relationship between 
the target and predictor is weak hence all categories of the target variable are 
receiving similar signals from the predictor making the predictor not a relevant 
variable for modelling.

Guided by that, the graph of average height per marital status shows very close average values 
hence very little differece is discernible. By this, height is likely not relevant for 
predicting marital status. Similar case can be made for weight.

Marked difference is noted in average age among various marital status. For instance,
average age for "Never married" is 19 years and that of "Married" is 43 years. With such 
large differences, it is likely to draw a decision boundary or a line for age with a large margin 
that clearly classifiers marital status hence age is deduced to be a relevant predictor.

"""

#%% Statistical test to determine if numeric predictors are relevant for predicting marital status

"""
While the bar graph visualization provides a pictorial way to understand whether a numeric predictor 
is relevant for predicting marital status, determining what constituent a significant noticiable 
difference between various categories is rather subjective. To introduce a more objective benchmark,
statistical test is required.

To determine if a predictor has a significant relationship with marital status hence a relevant predictor,
hypothesis is tested to determine whether there is a significant 
difference in predictors such as age, height and weight between various marital status. 
A significant difference suggests that the variable is a significant predictor of marital 
status hence have an influence that will improve the model. Univariate 
statistical methods are used to determine that.


### Feature selection: Is locale a relevant predictor of hits 

A factor that is considered in determining whether a categorical variable such as locale is a relevant 
predictor is the determination of variance of hits between the various categories of locale. By this, when 
hits significantly varies between the various categories of locale, it is likely to be a statistically 
significant predictor of hits. This notions applies to other categorical predictors such as
'agent_id', 'locale', 'day_of_week' and 'traffic_type'. For high cardinality predictors such as 'agent_id',
'entry_page' and 'path_id_set', it better to first 
treat them and reduce the classes to a manageable few before applying relevant statistical test.


In determining which statistical test to use, the assumptions required were tested to determine 
whether a parametric or non-parametric method of statistical test was appropriate. 
A parametric method such as Student t-test requires the data to be normally distributed and variance 
to be homogeneous for the various categories present in the predictor. 
When these assumptions are not captured by the data then a non-parametric method is appropriately used.


Both visualization and statistical methods are used to verify these assumptions. For a categorical 
variable, Boxplot is a good graphical technique to visulaize how hits varies within the various categories.
This is then supplemented with a levene test and bartlett test to statistically verify that variance 
depicted in by boxplot is indeed homogeneous. Barlett and levene method determines whether groups have homogeneous variance and levene is non-parametric method. 


The discussion is implemented in code for all low cardinal variables starting with locale as follows.



"""


#%%
def boxplot(data_to_plot: pd.DataFrame, x_colname: str, 
            y_colname: str,
            title: str = None
            ):
    if title is None:
        title = f'Distribution of {y_colname} among various {x_colname}'
        
    box_graph = (ggplot(data=data_to_plot, 
                        mapping=aes(x=x_colname, y=y_colname)
                        )
                    + geom_boxplot()
                    + coord_flip()
                    + ggtitle(title)
                )
    # the returned ggplot is printed to draw the graph which is not 
    # the case by default  when not printed
    return print(box_graph)
    
    
    
#%%
boxplot(data_to_plot=marital_status_df, x_colname='marital_status', y_colname='age_yrs')    


#%%

"""
From the boxplot of hits distribution among locales, there appears to be difference in how 
hits varies among the various locales and some data points are arguably outliers. 
A statistical test is undertaken to determine if hits in homogenous among the 
varous groups. Such a statistical test is premised on a hypothesis which is framed as 
follows: 


Null Hypothesis (H0): There is no statistically significant difference in variance of hits 
            between categories of a categorical predictor (locale)

Alternative Hypothesis (H1): There is statistically significant difference in variance of hits
            between categories of a categorical predictor (locale)
            
            
For all hypothesis test of homogeneity, this framework is assummed for each categorical predictor.

Both Levene test and Bartlett test are used to check homogeneity and implemented as follows:

""" 
 
    
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
## test of homogeneity of variance
for var in numeric_predictors:
    test_homogeneity(data=marital_status_df, target_var=var, 
                     predictor_var='marital_status'
                     )


"""
Given that null hypotheisis of homogeneity of variance was rejected for 
all, a non-parametric method will be used to test if various marital status 
have equal age, weight, and height. With marital status having more than 2 
categories, Kruskall Wallis will be used. While the p-value will be critical 
in drawing conclusion, it has already been deduced the differences in weight, 
height and even age among marital status are not really huge. Hence, the calculating 
the effect size is very important in this instance to know the extent of the 
difference. Kruskall Wallis is implemented as follows:



"""
#%%
class KruskallWallisTest(object):
    def __init__(self, data, group_var, variable):
        self.data = data
        self.group_var = group_var
        self.variable = variable
        
    def compute_kruskall_wallis_test(self):
        self.krus_res = pg.kruskal(data=self.data, 
                                   dv=self.variable, 
                                   between=self.group_var
                                   )
        return self.krus_res
    
    @property
    def effect_size(self):
        # effect_size for kruskal Wallis = (H -k + 1) / (n - k)
        # H = H-statistic from kruskal Wallis result
        # k = number of groups, n = number of observations
        k = self.data[self.group_var].nunique()

        n = self.data[self.variable].count()
        H = self.krus_res['H'][0]
        effect_size = (H -k + 1) / (n - k)
        print(f'Effect size: {self.group_var} vs {self.variable}')
        return effect_size
    
    
#%%
for var in numeric_predictors:
    krus = KruskallWallisTest(data = marital_status_df, 
                                group_var='marital_status', 
                                variable=var
                                )
    print(var)
    print(f'{krus.compute_kruskall_wallis_test()}')
    print(f'{krus.effect_size}\n')

"""
The result of the kruskal wallis test shows that all the predictors analyzed 
have a relationship with marital status based on the uncorrected p-value at 
5% significance level. The fact that the p-value being used is uncorrected for 
mutliple hypothesis testing means that we are prone making a Type I error. 

What is of a greater interest is the strength of the relationship between 
the predictors assessed and marital status. Age has an effect size of 0.67
which a moderate effect while height ans weight has 0.05 and 0.14 respectively 
to score a low effect. This actually corresponds to the bar graphs depicted earlier.
You is noted that for greater difference in age is seen between various marital 
status in the bar graph hence a higher effect size and the difference seen in weight among 
marital statuses on the bar graph is higher compared to that of height hence effect size being higher 
for weight compared to height. A possible implication of this on model is that when age 
is added to the predictors for modelling, the model will witness a higher improvement in prediction 
accuracy compared to weight and height in that order. Thus, for an instance where we want 
to select features to reduce overfitting and model complexity for computation, age will be 
selected at the expense of weight and height.  

"""

#%%

# Correlation analysis: Determining multicollinearity
""" 
In selecting numeric predictors for modelling, there is the need to prevent duplication 
of signal sources and redundant predictors need to be assessed and removed. A predictor 
is expected to provide a unique signal that contribute to making predictions. The problem 
with including redundant predictors in the model is that it becomes unstable, unnecassarily 
complex and even overfit the training data. 
Strong correlation between predictors
implies the predictors are supplying similar information to the algorithm.
Ever undertaking a regression analysis and 
released the coefficients of the predictors changes whenever a new predictor is added or removed
from the model? Chances are that some of the predictors are corelated. 
Multicollinearity is assessed to determine whether some of the numeric predictors are strongly 
correlated and if that is the case, for each correlated predictors, one one of them is 
selected for modelling.

Correlation analysis is undertaken on the numeric predictors to 
check for multicollinearity. Spearman method was used because the assumptions for 
a parametric method such as pearson are not met. 
"""
#%%

spearman_corr_matrix = marital_status_df[['weight','height','age_yrs']].corr(method='spearman')

#%%
import seaborn as sns
# Create a mask to hide the upper triangle
mask = np.zeros_like(spearman_corr_matrix)
mask[np.triu_indices_from(mask)] = True

# visualize correlation matrix
sns.heatmap(spearman_corr_matrix, mask=mask, cmap=sns.color_palette("GnBu_d"), 
            square=True, linewidths=.5, cbar_kws={"shrink": .5}
            )
plt.show()

#%%
"""
The spearman rho correlation result shows a general week correlation among 
the various predictors hence multicollinearity is absent. 

Based on all the analysis undertaken, age is selected as a predictor for modelling. 
Weight and height are not selected because they are less relevant for predicting 
marital status based on the bar plot. Even though the kruskall Wallis suggests the they are 
reletaed to marital status, the effect size are low and correspond to deduction from the 
bar plot visualization that they are not a good dsicriminator of marital status.
 
"""
#%%

### Feature selection for categorical features
#### Visualizing the relationship between categorical features and marital status
"""
The relationship between categorical predictors and marital status can be visualized using 
barplot that capture the count of the various classes of a categorical predictor in 
various marital status. For instance, for sex, the number of male and female per each 
marital status can be captured. Where male and female are equally distributed among the 
various marital status then sex is not related to marital status. The reverse where a particular
sex, say female is found more in a marital status example divorce will mean that sex is 
likely to be related and a good predictor.

The analysis is undertaken below:
"""


                         
#%%
(marital_status_df.groupby(by=['marital_status', 'father_in_hse'])
                            [['father_in_hse']].agg(func='count')
                            .rename(columns={'father_in_hse': 'total_count'})
                            .reset_index()
)

#%%
class CategoricalDataExplorer(object):
    def __init__(self, data, groupby_vars: str, vars_to_count: str):
        self.data = data
        self.groupby_vars = groupby_vars
        self.vars_to_count = vars_to_count
    def count_total_per_group(self, agg_method: str = 'count'):
        self.agg_data = (data.groupby(by=self.groupby_vars)
                                [[self.vars_to_count]].agg(func=agg_method)
                                #.rename_axis(columns={self.vars_to_count: 'total_count'})
                                #.reset_index()
                                .rename(columns={self.vars_to_count: 'total_count'})
                                .reset_index()
                    )
        return self.agg_data
    def plot_bar(self, xaxis_var: str = 'marital_status'):
        graph = (ggplot(self.agg_data, 
                aes(x=xaxis_var, y='total_count', fill=self.vars_to_count)
                ) + geom_col(stat='identity', position='dodge') 
                + theme_dark() 
                + ggtitle(f'Total count of {self.vars_to_count} per {xaxis_var}')
                + theme(axis_text_x=element_text(rotation=45, hjust=1))
                )
        print(graph)

#%%
categ_var =['father_in_hse', 'mother_in_hse', 'sex','attend_school', 'highest_edu']
     
for var in categ_var:
    cat_explr = CategoricalDataExplorer(data=marital_status_df,
                            groupby_vars=['marital_status', var],
                            vars_to_count=var
                            )
    cat_explr.count_total_per_group()
    cat_explr.plot_bar()


#%%
"""
While the visualization do provide some clues, statistical analysis will be a 
concrete stand on which to select relevant features that are related to marital status 
for prediction. 

For this, Chi squared test of independence will be used to assess whether 
the various categorical variables are related to marital status.

First contingency table is created between the categorical variable and marital status 
and based on the chi-squared test is computed to compare the observed and expected outcome
and decide on reject or failing to reject the null hypothesis of a categorical predictor 
being independent of marital status at 5% significant level. This is implemented as follows:



"""
#%%
from bioinfokit.analys import stat
class ChisquaredComputer(object):
    def __init__(self, data, y_var, x_var):
        self.data = data
        self.y_var = y_var
        self.x_var = x_var
        
    def compute_contingency_table(self):
        self.contengency_table = pd.crosstab(self.data[self.y_var], 
                                             self.data[self.x_var]
                                            )

        return self.contengency_table
    def compute_chisquared_test_of_independence(self):
        chi_square_result = stat()
        chi_square_result.chisq(df=self.contengency_table)
        return print(f'Relationship between {self.y_var} and {self.x_var} \n {chi_square_result.summary}')


#%%
for var in categ_var:
    chisquared = ChisquaredComputer(data=marital_status_df, 
                                    x_var=var, 
                                    y_var='marital_status'
                                    )
        
    chisquared.compute_contingency_table()

    chisquared.compute_chisquared_test_of_independence()


"""
From the chi square analysis all the categorical varibles are relevant for modelling 
hence selected
"""

#%%
"""
### Using insights gained from exploratory analysis to inform modelling approach

The findings of a discrete target variable informs the decisio to use a supervised classifcation a
algorithm.


non-linear relationship between 
the predictors and target variable, presence of outliers,
and sizeable missing values suggest that a non-parametric
model that handles non-linear relationship,
outliers and missing values will be appropriate for the task. Also, the fact missing data has already been handled through feature engineering and selection was also considered.
 
Moreover, as identified in the objectives, the focus of the task is to achieve good precision with a minimum requirement of being better than random guesses rather than interpretability of the model and this informed a decision to choose an algorithm that satisfy these needs. 


On the basis of the findings from the exploratory analysis, a model 
that employs decision tree will be used. Hence, Random Forest Regressor algorithm from the sklearn library is implemented.

Given that the target variable is known and continuous, supervised regression is used for the task. 


From the exploratory analysis, the filter-based selection method resulted in the selection of 'neighbourhood_group', 'room_type', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count' and 'availability_365' as predictors for modelling. In addition, 
'review_is_available' is produced from feature engineering to be added to predictors for modelling.
       
"""








#%%
pg.anova(data=marital_status_df, dv='weight_outlier_imputed', between='marital_status')
       
    
#%%

marital_status_df[['weight_outlier_imputed', 
                    'height_outlier_imputed', 
                    'age_yrs_outlier_imputed'
                    ]].corr(method='spearman')


    
    
#%%

for var in ['weight_outlier_imputed', 'height_outlier_imputed', 'age_yrs_outlier_imputed']:
    test_homogeneity(data=marital_status_df, target_var=var, predictor_var='marital_status')    


#%%

for var in ['weight_outlier_imputed', 'height_outlier_imputed', 'age_yrs_outlier_imputed']:
    krus = KruskallWallisTest(data = marital_status_df, 
                                group_var='marital_status', 
                                variable=var
                                )
    print(var)
    print(f'{krus.compute_kruskall_wallis_test()}')
    print(f'{krus.effect_size}\n')


#%%

krus = KruskallWallisTest(data = marital_status_df, 
                         group_var='marital_status', 
                         variable='age_yrs_outlier_imputed'
                        )
    
krus.compute_kruskall_wallis_test()
krus.effect_size







#%%
import pingouin as pg
pg.kruskal(data=marital_status_df, dv='weight', between='marital_status')

#%%

krus_res = pg.kruskal(data=marital_status_df, dv='weight', between='marital_status')


# eta2[H] = (H - k + 1)/(n - k); where H is the value obtained in the Kruskal-Wallis test; k is the number of groups; n is the total number of observations.

#%%

k = marital_status_df['marital_status'].nunique()

n = marital_status_df['weight'].count()

#%%
H = krus_res['H'][0]

#%%

(H -k + 1) / (n - k)

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










