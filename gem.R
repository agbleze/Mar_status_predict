library(haven)


gem2018 = '/Volumes/Elements/GEM_2018_APS_Global_Individual_Level_Data.sav'



haven::read_sav(gem2018)

haven::read_spss(gem2018)


df = haven::read_sav('/Volumes/Elements/DOCUMENTS/GEM/GEM 2016 APS Global - Individual Level Data.sav')


colnames(df)

df["DISCENyy"]  # discontinued business in the last 12 month

# INDSUPyy  Individual perception of entrepreneurship index

# CULSUPyy Cultural support for enterpreneurship index

# EXIT_CTD   Discontinued a business in the past 12 months but business continued

# EXIT_ENT  Discontinued a business in the past 12 months (include business that were continued)

# FRFAILOP   Fear of failure (in 18-64 sample perceiving good opportunity to start business)

