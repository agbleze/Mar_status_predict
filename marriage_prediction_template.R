file = "/Volumes/Elements/Data_migrant_health/g7sec1_5 Migrant_health data.sav"


migrant_health <- haven::read_sav(file)


migrant_health$s1q6

table(migrant_health$s1q6)


health = "/Volumes/Elements/Data_migrant_health/g7sec3b_Health_Insurance.sav"

health_insur = haven::read_sav(health)

g7sec1 <- "/Volumes/Elements/Data_migrant_health/g7sec1.sav"


df_g7sec1 <- haven::read_sav(g7sec1)



####### Predicting age at which people will get married or leave with partner  ###########
### 9 At what age did (NAME) first get married or start living with a partner?


#######  Predicting the marital status of people   ########

## Target variable

# s1q6  What is (NAME’S) present marital status?



## predictors

## age_yrs
## age_mths
## s7jq1  weight
## s7jq0 was name measured
## s7jq2 mode of measuring height
## s7jq3 Height
## loc2 urbrur
## s1q22 Number of months away from houshold <<  22 For how many months during the past 12 months has (NAME) been continuously away from this household? (IF 6 MONTHS OR LESS >>  24)

## s1q14 father live in household  <<  14 Does (NAME’S) father live in this household?
## s1q1b educational level <<  What is the highest level of education (NAME) has attained?


## s1q2 sex of individual
## s1q18  Mother live in household  <<  18 Does (NAME’S) mother live in this household?

## s1q5y Age in years  <<  5 How old is (NAME)? YEARS AND MONTHS IF UNDER 5 YEARS, OTHERWISE YEARS ONLY


## s1q10 What is (NAME’S) religious denomination?

## s2aq1b  << 1b What is the highest level of education (NAME) has attained?
## s2aq1  << 1 Has (NAME) ever attended school?



table(health_insur$s3bq1)
health_insur$s3bq1



