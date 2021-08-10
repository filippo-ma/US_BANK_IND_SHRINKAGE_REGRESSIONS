import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import LeaveOneOut

from dates_configuration import *




def open_my_csv(csv_file):
    
    data = pd.read_csv(csv_file)
    df = pd.DataFrame(data).set_index(['Unnamed: 0', 'Unnamed: 1']).rename_axis(index={'Unnamed: 0': '', 'Unnamed: 1': ''})
    
    return df



def get_ttm(df_object, df_object_date, half):
    """
    get the trailing twelve months figure for the chosen fundamental for each company.
    df_object = df variable to be processed, ex: income_statement_quarter['revenue'].
    df_object_date = df dates, ex: income_statement_quarter.date.
    half = selected semester at which time ttm must be computed, ex: 1=last sem., 2=2nd last sem.,...
    """
    df_dates = pd.DataFrame(df_object_date)
    df_values = pd.DataFrame(df_object)
    
    df_dates.columns = ['date']

    df = pd.concat([df_dates, df_values], axis=1)

    df_grouped = df.groupby(level=0)
    
    # Take only the latest 4 dates (for each half). Then, filter out any groups without exactly 4 qtrs of data.
    df_1 = df_grouped.head(4)
    df_2 = df_grouped.head(6).groupby(level=0).tail(4)
    df_3 = df_grouped.head(8).groupby(level=0).tail(4)
    df_4 = df_grouped.head(10).groupby(level=0).tail(4)
    df_5 = df_grouped.head(12).groupby(level=0).tail(4)
    df_6 = df_grouped.head(14).groupby(level=0).tail(4)
    df_7 = df_grouped.head(16).groupby(level=0).tail(4)
    df_8 = df_grouped.head(18).groupby(level=0).tail(4)
    exact_df1 = df_1 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df2 = df_2 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df3 = df_3 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df4 = df_4 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df5 = df_5 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df6 = df_6 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df7 = df_7 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df8 = df_8 #.groupby(level=0).filter(lambda group: group.date.size == 4)


    # values sums where there are exactly 4 qtrs to get TTM.
    ttm_df_last_half = exact_df1.groupby(level=0).sum(min_count=4)
    ttm_df_2nd_last_half = exact_df2.groupby(level=0).sum(min_count=4)
    ttm_df_3rd_last_half = exact_df3.groupby(level=0).sum(min_count=4)
    ttm_df_4th_last_half = exact_df4.groupby(level=0).sum(min_count=4)
    ttm_df_5th_last_half = exact_df5.groupby(level=0).sum(min_count=4)
    ttm_df_6th_last_half = exact_df6.groupby(level=0).sum(min_count=4)
    ttm_df_7th_last_half = exact_df7.groupby(level=0).sum(min_count=4)
    ttm_df_8th_last_half = exact_df8.groupby(level=0).sum(min_count=4)


    if half == 1:
        return ttm_df_last_half
    elif half == 2:
        return ttm_df_2nd_last_half
    elif half == 3: 
        return ttm_df_3rd_last_half
    elif half == 4:
        return ttm_df_4th_last_half
    elif half == 5:
        return ttm_df_5th_last_half
    elif half == 6:
        return ttm_df_6th_last_half
    elif half == 7:
        return ttm_df_7th_last_half    
    else:
        return ttm_df_8th_last_half

def get_ttm_avg(df_object, df_object_date, half):

    """
    get the trailing twelve months figure (average) for the chosen fundamental for each company.
    df_object = df variable to be processed, ex: balance_sheet_statement_quarter['totalAssets'].
    df_object_date = df dates, ex: balance_sheet_statement_quarter_df.date.
    half = selected semester at which time ttm must be computed, ex: 1=last sem., 2=2nd last sem.,...
    """
    df_dates = pd.DataFrame(df_object_date)
    df_values = pd.DataFrame(df_object)
    
    df_dates.columns = ['date']

    df = pd.concat([df_dates, df_values], axis=1)

    df_grouped = df.groupby(level=0)
    
    # Take only the latest 4 dates (for each half). Then, filter out any groups without exactly 4 qtrs of data.
    df_1 = df_grouped.head(4)
    df_2 = df_grouped.head(6).groupby(level=0).tail(4)
    df_3 = df_grouped.head(8).groupby(level=0).tail(4)
    df_4 = df_grouped.head(10).groupby(level=0).tail(4)
    df_5 = df_grouped.head(12).groupby(level=0).tail(4)
    df_6 = df_grouped.head(14).groupby(level=0).tail(4)
    df_7 = df_grouped.head(16).groupby(level=0).tail(4)
    df_8 = df_grouped.head(18).groupby(level=0).tail(4)
    exact_df1 = df_1 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df2 = df_2 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df3 = df_3 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df4 = df_4 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df5 = df_5 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df6 = df_6 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df7 = df_7 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df8 = df_8 #.groupby(level=0).filter(lambda group: group.date.size == 4)


    # values avg where there are 4 qtrs to get TTM.
    ttm_df_last_half = exact_df1.groupby(level=0).aggregate(np.mean)
    ttm_df_2nd_last_half = exact_df2.groupby(level=0).aggregate(np.mean)
    ttm_df_3rd_last_half = exact_df3.groupby(level=0).aggregate(np.mean)
    ttm_df_4th_last_half = exact_df4.groupby(level=0).aggregate(np.mean)
    ttm_df_5th_last_half = exact_df5.groupby(level=0).aggregate(np.mean)
    ttm_df_6th_last_half = exact_df6.groupby(level=0).aggregate(np.mean)
    ttm_df_7th_last_half = exact_df7.groupby(level=0).aggregate(np.mean)
    ttm_df_8th_last_half = exact_df8.groupby(level=0).aggregate(np.mean)
    

    if half == 1:
        return ttm_df_last_half
    elif half == 2:
        return ttm_df_2nd_last_half
    elif half == 3: 
        return ttm_df_3rd_last_half
    elif half == 4:
        return ttm_df_4th_last_half
    elif half == 5:
        return ttm_df_5th_last_half
    elif half == 6:
        return ttm_df_6th_last_half
    elif half == 7:
        return ttm_df_7th_last_half    
    else:
        return ttm_df_8th_last_half

def get_ttm_dep(df_object, df_object_date, half):

    """
    get the trailing twelve months figure (average) for the chosen fundamental for each company.
    df_object = df variable to be processed, ex: balance_sheet_statement_quarter_df['assets'].
    df_object_date = df dates, ex: balance_sheet_statement_quarter_df.date.
    half = selected semester at which time ttm must be computed, ex: 1=last sem., 2=2nd last sem.,...
    """
    df_dates = pd.DataFrame(df_object_date)
    df_values = pd.DataFrame(df_object)
    
    df_dates.columns = ['date']

    df = pd.concat([df_dates, df_values], axis=1)

    df_grouped = df.groupby(level=0)
    
    # Take only the latest 4 dates (for each half). Then, filter out any groups without exactly 4 qtrs of data.
    df_1 = df_grouped.head(4)
    df_2 = df_grouped.head(6).groupby(level=0).tail(4)
    df_3 = df_grouped.head(8).groupby(level=0).tail(4)
    df_4 = df_grouped.head(10).groupby(level=0).tail(4)
    df_5 = df_grouped.head(12).groupby(level=0).tail(4)
    df_6 = df_grouped.head(14).groupby(level=0).tail(4)
    df_7 = df_grouped.head(16).groupby(level=0).tail(4)
    df_8 = df_grouped.head(18).groupby(level=0).tail(4)
    exact_df1 = df_1 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df2 = df_2 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df3 = df_3 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df4 = df_4 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df5 = df_5 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df6 = df_6 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df7 = df_7 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df8 = df_8 #.groupby(level=0).filter(lambda group: group.date.size == 4)


    # values avg where there are 4 qtrs to get TTM.
    ttm_df_last_half = exact_df1.groupby(level=0).mean()
    ttm_df_2nd_last_half = exact_df2.groupby(level=0).mean()
    ttm_df_3rd_last_half = exact_df3.groupby(level=0).mean()
    ttm_df_4th_last_half = exact_df4.groupby(level=0).mean()
    ttm_df_5th_last_half = exact_df5.groupby(level=0).mean()
    ttm_df_6th_last_half = exact_df6.groupby(level=0).mean()
    ttm_df_7th_last_half = exact_df7.groupby(level=0).mean()
    ttm_df_8th_last_half = exact_df8.groupby(level=0).mean()
    

    if half == 1:
        return ttm_df_last_half
    elif half == 2:
        return ttm_df_2nd_last_half
    elif half == 3: 
        return ttm_df_3rd_last_half
    elif half == 4:
        return ttm_df_4th_last_half
    elif half == 5:
        return ttm_df_5th_last_half
    elif half == 6:
        return ttm_df_6th_last_half
    elif half == 7:
        return ttm_df_7th_last_half    
    else:
        return ttm_df_8th_last_half

def compute_ttm_change(df, df_object, df_object_date, half):
    """
    compute the last twelve months pct change of for the chosen fundamental for each company.
    df = ex: income_statement_df (pandas dataframe).
    df_object = df variable to be processed, ex: 'revenue' (dtype: string) (must be a column name of df).
    df_object_date = df dates, ex: income_statement_df.date
    half = selected semester at which time ttm must be computed, ex: 1=last sem., 2=2nd last sem.,...
    """
    df_dates = pd.DataFrame(df_object_date)
    df_values = pd.DataFrame(df['{}'.format(df_object)])
    
    df_dates.columns = ['date']

    df = pd.concat([df_dates, df_values], axis=1)

    df_grouped = df.groupby(level=0)
    
    # Take only the latest 4 dates (for each half). Then, filter out any groups without exactly 4 qtrs of data.
    df_1 = df_grouped.head(4)
    df_2 = df_grouped.head(6).groupby(level=0).tail(4)
    df_3 = df_grouped.head(8).groupby(level=0).tail(4)
    df_4 = df_grouped.head(10).groupby(level=0).tail(4)
    df_5 = df_grouped.head(12).groupby(level=0).tail(4)
    df_6 = df_grouped.head(14).groupby(level=0).tail(4)
    df_7 = df_grouped.head(16).groupby(level=0).tail(4)
    df_8 = df_grouped.head(18).groupby(level=0).tail(4)
    exact_df1 = df_1 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df2 = df_2 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df3 = df_3 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df4 = df_4 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df5 = df_5 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df6 = df_6 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df7 = df_7 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df8 = df_8 #.groupby(level=0).filter(lambda group: group.date.size == 4)


    # values sums where there are exactly 4 qtrs to get TTM.
    ttm_df_last_half = exact_df1.groupby(level=0).sum(min_count=4)
    ttm_df_2nd_last_half = exact_df2.groupby(level=0).sum(min_count=4)
    ttm_df_3rd_last_half = exact_df3.groupby(level=0).sum(min_count=4)
    ttm_df_4th_last_half = exact_df4.groupby(level=0).sum(min_count=4)
    ttm_df_5th_last_half = exact_df5.groupby(level=0).sum(min_count=4)
    ttm_df_6th_last_half = exact_df6.groupby(level=0).sum(min_count=4)
    ttm_df_7th_last_half = exact_df7.groupby(level=0).sum(min_count=4)
    ttm_df_8th_last_half = exact_df8.groupby(level=0).sum(min_count=4)

    
    # compute pct change
    ttm_change_1 = (ttm_df_last_half.values - ttm_df_3rd_last_half.values) / np.abs(ttm_df_3rd_last_half.values)
    ttm_change_2 = (ttm_df_2nd_last_half.values - ttm_df_4th_last_half.values) / np.abs(ttm_df_4th_last_half.values)
    ttm_change_3 = (ttm_df_3rd_last_half.values - ttm_df_5th_last_half.values) / np.abs(ttm_df_5th_last_half.values)
    ttm_change_4 = (ttm_df_4th_last_half.values - ttm_df_6th_last_half.values) / np.abs(ttm_df_6th_last_half.values)
    ttm_change_5 = (ttm_df_5th_last_half.values - ttm_df_7th_last_half.values) / np.abs(ttm_df_7th_last_half.values)
    ttm_change_6 = (ttm_df_6th_last_half.values - ttm_df_8th_last_half.values) / np.abs(ttm_df_8th_last_half.values)


    if half == 1:
        return pd.DataFrame(ttm_change_1, columns=['{} ttmChange'.format(df_object)], index=ttm_df_last_half.index)
    elif half == 2:
        return pd.DataFrame(ttm_change_2, columns=['{} ttmChange'.format(df_object)], index=ttm_df_2nd_last_half.index)
    elif half == 3:
        return pd.DataFrame(ttm_change_3, columns=['{} ttmChange'.format(df_object)], index=ttm_df_3rd_last_half.index)
    elif half == 4:
        return pd.DataFrame(ttm_change_4, columns=['{} ttmChange'.format(df_object)], index=ttm_df_4th_last_half.index)
    elif half == 5:
        return pd.DataFrame(ttm_change_5, columns=['{} ttmChange'.format(df_object)], index=ttm_df_5th_last_half.index) 
    else:
        return pd.DataFrame(ttm_change_6, columns=['{} ttmChange'.format(df_object)], index=ttm_df_6th_last_half.index)   

def delta_ttm(df, df_object, df_object_date, half):

    df_dates = pd.DataFrame(df_object_date)
    df_values = pd.DataFrame(df['{}'.format(df_object)])
    
    df_dates.columns = ['date']

    df = pd.concat([df_dates, df_values], axis=1)

    df_grouped = df.groupby(level=0)
    
    # Take only the latest 4 dates (for each half). Then, filter out any groups without exactly 4 qtrs of data.
    df_1 = df_grouped.head(4)
    df_2 = df_grouped.head(6).groupby(level=0).tail(4)
    df_3 = df_grouped.head(8).groupby(level=0).tail(4)
    df_4 = df_grouped.head(10).groupby(level=0).tail(4)
    df_5 = df_grouped.head(12).groupby(level=0).tail(4)
    df_6 = df_grouped.head(14).groupby(level=0).tail(4)
    df_7 = df_grouped.head(16).groupby(level=0).tail(4)
    df_8 = df_grouped.head(18).groupby(level=0).tail(4)
    df_9 = df_grouped.head(20).groupby(level=0).tail(4)
    df_10 = df_grouped.head(22).groupby(level=0).tail(4)
    exact_df1 = df_1 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df2 = df_2 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df3 = df_3 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df4 = df_4 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df5 = df_5 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df6 = df_6 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df7 = df_7 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df8 = df_8 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df9 = df_9 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df10 = df_10 #.groupby(level=0).filter(lambda group: group.date.size == 4)


    # values sums where there are exactly 4 qtrs to get TTM.
    ttm_df_last_half = exact_df1.groupby(level=0).sum(min_count=4)
    ttm_df_2nd_last_half = exact_df2.groupby(level=0).sum(min_count=4)
    ttm_df_3rd_last_half = exact_df3.groupby(level=0).sum(min_count=4)
    ttm_df_4th_last_half = exact_df4.groupby(level=0).sum(min_count=4)
    ttm_df_5th_last_half = exact_df5.groupby(level=0).sum(min_count=4)
    ttm_df_6th_last_half = exact_df6.groupby(level=0).sum(min_count=4)
    ttm_df_7th_last_half = exact_df7.groupby(level=0).sum(min_count=4)
    ttm_df_8th_last_half = exact_df8.groupby(level=0).sum(min_count=4)
    ttm_df_9th_last_half = exact_df9.groupby(level=0).sum(min_count=4)
    ttm_df_10th_last_half = exact_df10.groupby(level=0).sum(min_count=4)

    
    # compute delta
    ttm_change_1 = (ttm_df_last_half.values - ttm_df_3rd_last_half.values) 
    ttm_change_2 = (ttm_df_2nd_last_half.values - ttm_df_4th_last_half.values)
    ttm_change_3 = (ttm_df_3rd_last_half.values - ttm_df_5th_last_half.values)
    ttm_change_4 = (ttm_df_4th_last_half.values - ttm_df_6th_last_half.values)
    ttm_change_5 = (ttm_df_5th_last_half.values - ttm_df_7th_last_half.values)
    ttm_change_6 = (ttm_df_6th_last_half.values - ttm_df_8th_last_half.values)
    ttm_change_7 = (ttm_df_7th_last_half.values - ttm_df_9th_last_half.values)


    if half == 1:
        return pd.DataFrame(ttm_change_1, columns=['{} delta ttm'.format(df_object)], index=ttm_df_last_half.index)
    elif half == 2:
        return pd.DataFrame(ttm_change_2, columns=['{} delta ttm'.format(df_object)], index=ttm_df_2nd_last_half.index)
    elif half == 3:
        return pd.DataFrame(ttm_change_3, columns=['{} delta ttm'.format(df_object)], index=ttm_df_3rd_last_half.index)
    elif half == 4:
        return pd.DataFrame(ttm_change_4, columns=['{} delta ttm'.format(df_object)], index=ttm_df_4th_last_half.index)
    elif half == 5:
        return pd.DataFrame(ttm_change_5, columns=['{} delta ttm'.format(df_object)], index=ttm_df_5th_last_half.index) 
    elif half == 6:
        return pd.DataFrame(ttm_change_6, columns=['{} delta ttm'.format(df_object)], index=ttm_df_6th_last_half.index)
    else:
        return pd.DataFrame(ttm_change_7, columns=['{} delta ttm'.format(df_object)], index=ttm_df_7th_last_half.index)

def compute_avg_ttm_change(df, df_object, df_object_date, half):

    """
    compute the last twelve months pct change of for the chosen fundamental (avg ttm) for each company.
    df = ex: income_statement_df (pandas dataframe).
    df_object = df variable to be processed, ex: 'revenue' (dtype: string) (must be a column name of df).
    df_object_date = df dates, ex: income_statement_df.date
    half = selected semester at which time ttm must be computed, ex: 1=last sem., 2=2nd last sem.,...
    """
    df_dates = pd.DataFrame(df_object_date)
    df_values = pd.DataFrame(df['{}'.format(df_object)])
    
    df_dates.columns = ['date']

    df = pd.concat([df_dates, df_values], axis=1)

    df_grouped = df.groupby(level=0)
    
    # Take only the latest 4 dates (for each half). Then, filter out any groups without exactly 4 qtrs of data.
    df_1 = df_grouped.head(4)   
    df_2 = df_grouped.head(6).groupby(level=0).tail(4)
    df_3 = df_grouped.head(8).groupby(level=0).tail(4)
    df_4 = df_grouped.head(10).groupby(level=0).tail(4)
    df_5 = df_grouped.head(12).groupby(level=0).tail(4)
    df_6 = df_grouped.head(14).groupby(level=0).tail(4)
    df_7 = df_grouped.head(16).groupby(level=0).tail(4)
    df_8 = df_grouped.head(18).groupby(level=0).tail(4)
    df_9 = df_grouped.head(20).groupby(level=0).tail(4)
    df_10 = df_grouped.head(22).groupby(level=0).tail(4)
    exact_df1 = df_1 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df2 = df_2 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df3 = df_3 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df4 = df_4 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df5 = df_5 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df6 = df_6 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df7 = df_7 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df8 = df_8 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df9 = df_9 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df10 = df_10 #.groupby(level=0).filter(lambda group: group.date.size == 4)


    # values sums where there are exactly 4 qtrs to get TTM.
    ttm_df_last_half = exact_df1.groupby(level=0).aggregate(np.mean)
    ttm_df_2nd_last_half = exact_df2.groupby(level=0).aggregate(np.mean)
    ttm_df_3rd_last_half = exact_df3.groupby(level=0).aggregate(np.mean)
    ttm_df_4th_last_half = exact_df4.groupby(level=0).aggregate(np.mean)
    ttm_df_5th_last_half = exact_df5.groupby(level=0).aggregate(np.mean)
    ttm_df_6th_last_half = exact_df6.groupby(level=0).aggregate(np.mean)
    ttm_df_7th_last_half = exact_df7.groupby(level=0).aggregate(np.mean)
    ttm_df_8th_last_half = exact_df8.groupby(level=0).aggregate(np.mean)
    ttm_df_9th_last_half = exact_df9.groupby(level=0).aggregate(np.mean)
    ttm_df_10th_last_half = exact_df10.groupby(level=0).aggregate(np.mean)
    
    # compute delta ttm
    ttm_change_1 = (ttm_df_last_half.values - ttm_df_3rd_last_half.values) / np.abs(ttm_df_3rd_last_half.values)
    ttm_change_2 = (ttm_df_2nd_last_half.values - ttm_df_4th_last_half.values) / np.abs(ttm_df_4th_last_half.values)
    ttm_change_3 = (ttm_df_3rd_last_half.values - ttm_df_5th_last_half.values) / np.abs(ttm_df_5th_last_half.values)
    ttm_change_4 = (ttm_df_4th_last_half.values - ttm_df_6th_last_half.values) / np.abs(ttm_df_6th_last_half.values)
    ttm_change_5 = (ttm_df_5th_last_half.values - ttm_df_7th_last_half.values) / np.abs(ttm_df_7th_last_half.values)
    ttm_change_6 = (ttm_df_6th_last_half.values - ttm_df_8th_last_half.values) / np.abs(ttm_df_8th_last_half.values)
    ttm_change_7 = (ttm_df_7th_last_half.values - ttm_df_9th_last_half.values) / np.abs(ttm_df_9th_last_half.values)


    if half == 1:
        return pd.DataFrame(ttm_change_1, columns=['{} ttmChange'.format(df_object)], index=ttm_df_last_half.index)
    elif half == 2:
        return pd.DataFrame(ttm_change_2, columns=['{} ttmChange'.format(df_object)], index=ttm_df_2nd_last_half.index)
    elif half == 3:
        return pd.DataFrame(ttm_change_3, columns=['{} ttmChange'.format(df_object)], index=ttm_df_3rd_last_half.index)
    elif half == 4:
        return pd.DataFrame(ttm_change_4, columns=['{} ttmChange'.format(df_object)], index=ttm_df_4th_last_half.index)
    elif half == 5:
        return pd.DataFrame(ttm_change_5, columns=['{} ttmChange'.format(df_object)], index=ttm_df_5th_last_half.index) 
    elif half == 6:
        return pd.DataFrame(ttm_change_6, columns=['{} ttmChange'.format(df_object)], index=ttm_df_6th_last_half.index)
    else:
        return pd.DataFrame(ttm_change_7, columns=['{} ttmChange'.format(df_object)], index=ttm_df_7th_last_half.index)    

def delta_ttm_avg(df, df_object, df_object_date, half):

    df_dates = pd.DataFrame(df_object_date)
    df_values = pd.DataFrame(df['{}'.format(df_object)])
    
    df_dates.columns = ['date']

    df = pd.concat([df_dates, df_values], axis=1)

    df_grouped = df.groupby(level=0)
    
    # Take only the latest 4 dates (for each half). Then, filter out any groups without exactly 4 qtrs of data.
    df_1 = df_grouped.head(4)
    df_2 = df_grouped.head(6).groupby(level=0).tail(4)
    df_3 = df_grouped.head(8).groupby(level=0).tail(4)
    df_4 = df_grouped.head(10).groupby(level=0).tail(4)
    df_5 = df_grouped.head(12).groupby(level=0).tail(4)
    df_6 = df_grouped.head(14).groupby(level=0).tail(4)
    df_7 = df_grouped.head(16).groupby(level=0).tail(4)
    df_8 = df_grouped.head(18).groupby(level=0).tail(4)
    df_9 = df_grouped.head(20).groupby(level=0).tail(4)
    df_10 = df_grouped.head(22).groupby(level=0).tail(4)
    exact_df1 = df_1 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df2 = df_2 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df3 = df_3 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df4 = df_4 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df5 = df_5 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df6 = df_6 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df7 = df_7 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df8 = df_8 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df9 = df_9 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df10 = df_10 #.groupby(level=0).filter(lambda group: group.date.size == 4)


    # values sums where there are exactly 4 qtrs to get TTM.
    ttm_df_last_half = exact_df1.groupby(level=0).aggregate(np.mean)
    ttm_df_2nd_last_half = exact_df2.groupby(level=0).aggregate(np.mean)
    ttm_df_3rd_last_half = exact_df3.groupby(level=0).aggregate(np.mean)
    ttm_df_4th_last_half = exact_df4.groupby(level=0).aggregate(np.mean)
    ttm_df_5th_last_half = exact_df5.groupby(level=0).aggregate(np.mean)
    ttm_df_6th_last_half = exact_df6.groupby(level=0).aggregate(np.mean)
    ttm_df_7th_last_half = exact_df7.groupby(level=0).aggregate(np.mean)
    ttm_df_8th_last_half = exact_df8.groupby(level=0).aggregate(np.mean)
    ttm_df_9th_last_half = exact_df9.groupby(level=0).aggregate(np.mean)
    ttm_df_10th_last_half = exact_df10.groupby(level=0).aggregate(np.mean)

    
    # compute delta ttm
    ttm_change_1 = (ttm_df_last_half.values - ttm_df_3rd_last_half.values) 
    ttm_change_2 = (ttm_df_2nd_last_half.values - ttm_df_4th_last_half.values)
    ttm_change_3 = (ttm_df_3rd_last_half.values - ttm_df_5th_last_half.values)
    ttm_change_4 = (ttm_df_4th_last_half.values - ttm_df_6th_last_half.values)
    ttm_change_5 = (ttm_df_5th_last_half.values - ttm_df_7th_last_half.values)
    ttm_change_6 = (ttm_df_6th_last_half.values - ttm_df_8th_last_half.values)
    ttm_change_7 = (ttm_df_7th_last_half.values - ttm_df_9th_last_half.values)


    if half == 1:
        return pd.DataFrame(ttm_change_1, columns=['{} delta ttm'.format(df_object)], index=ttm_df_last_half.index)
    elif half == 2:
        return pd.DataFrame(ttm_change_2, columns=['{} delta ttm'.format(df_object)], index=ttm_df_2nd_last_half.index)
    elif half == 3:
        return pd.DataFrame(ttm_change_3, columns=['{} delta ttm'.format(df_object)], index=ttm_df_3rd_last_half.index)
    elif half == 4:
        return pd.DataFrame(ttm_change_4, columns=['{} delta ttm'.format(df_object)], index=ttm_df_4th_last_half.index)
    elif half == 5:
        return pd.DataFrame(ttm_change_5, columns=['{} delta ttm'.format(df_object)], index=ttm_df_5th_last_half.index) 
    elif half == 6:
        return pd.DataFrame(ttm_change_6, columns=['{} delta ttm'.format(df_object)], index=ttm_df_6th_last_half.index)
    else:
        return pd.DataFrame(ttm_change_7, columns=['{} delta ttm'.format(df_object)], index=ttm_df_7th_last_half.index)

def compute_ttm_change_dep(df, df_object, df_object_date, half):

    """
    compute the last twelve months pct change of for the chosen fundamental (avg ttm) for each company.
    df = ex: income_statement_df (pandas dataframe).
    df_object = df variable to be processed, ex: 'revenue' (dtype: string) (must be a column name of df).
    df_object_date = df dates, ex: income_statement_df.date
    half = selected semester at which time ttm must be computed, ex: 1=last sem., 2=2nd last sem.,...
    """
    df_dates = pd.DataFrame(df_object_date)
    df_values = pd.DataFrame(df['{}'.format(df_object)])
    
    df_dates.columns = ['date']

    df = pd.concat([df_dates, df_values], axis=1)

    df_grouped = df.groupby(level=0)
    
    # Take only the latest 4 dates (for each half). Then, filter out any groups without exactly 4 qtrs of data.
    df_1 = df_grouped.head(4)
    df_2 = df_grouped.head(6).groupby(level=0).tail(4)
    df_3 = df_grouped.head(8).groupby(level=0).tail(4)
    df_4 = df_grouped.head(10).groupby(level=0).tail(4)
    df_5 = df_grouped.head(12).groupby(level=0).tail(4)
    df_6 = df_grouped.head(14).groupby(level=0).tail(4)
    df_7 = df_grouped.head(16).groupby(level=0).tail(4)
    df_8 = df_grouped.head(18).groupby(level=0).tail(4)
    exact_df1 = df_1 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df2 = df_2 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df3 = df_3 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df4 = df_4 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df5 = df_5 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df6 = df_6 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df7 = df_7 #.groupby(level=0).filter(lambda group: group.date.size == 4)
    exact_df8 = df_8 #.groupby(level=0).filter(lambda group: group.date.size == 4)


    # values sums where there are exactly 4 qtrs to get TTM.
    ttm_df_last_half = exact_df1.groupby(level=0).mean()
    ttm_df_2nd_last_half = exact_df2.groupby(level=0).mean()
    ttm_df_3rd_last_half = exact_df3.groupby(level=0).mean()
    ttm_df_4th_last_half = exact_df4.groupby(level=0).mean()
    ttm_df_5th_last_half = exact_df5.groupby(level=0).mean()
    ttm_df_6th_last_half = exact_df6.groupby(level=0).mean()
    ttm_df_7th_last_half = exact_df7.groupby(level=0).mean()
    ttm_df_8th_last_half = exact_df8.groupby(level=0).mean()

    
    # compute pct change
    ttm_change_1 = (ttm_df_last_half.values - ttm_df_3rd_last_half.values) / np.abs(ttm_df_3rd_last_half.values)
    ttm_change_2 = (ttm_df_2nd_last_half.values - ttm_df_4th_last_half.values) / np.abs(ttm_df_4th_last_half.values)
    ttm_change_3 = (ttm_df_3rd_last_half.values - ttm_df_5th_last_half.values) / np.abs(ttm_df_5th_last_half.values)
    ttm_change_4 = (ttm_df_4th_last_half.values - ttm_df_6th_last_half.values) / np.abs(ttm_df_6th_last_half.values)
    ttm_change_5 = (ttm_df_5th_last_half.values - ttm_df_7th_last_half.values) / np.abs(ttm_df_7th_last_half.values)
    ttm_change_6 = (ttm_df_6th_last_half.values - ttm_df_8th_last_half.values) / np.abs(ttm_df_8th_last_half.values)


    if half == 1:
        return pd.DataFrame(ttm_change_1, columns=['{} ttmChange'.format(df_object)], index=ttm_df_last_half.index)
    elif half == 2:
        return pd.DataFrame(ttm_change_2, columns=['{} ttmChange'.format(df_object)], index=ttm_df_2nd_last_half.index)
    elif half == 3:
        return pd.DataFrame(ttm_change_3, columns=['{} ttmChange'.format(df_object)], index=ttm_df_3rd_last_half.index)
    elif half == 4:
        return pd.DataFrame(ttm_change_4, columns=['{} ttmChange'.format(df_object)], index=ttm_df_4th_last_half.index)
    elif half == 5:
        return pd.DataFrame(ttm_change_5, columns=['{} ttmChange'.format(df_object)], index=ttm_df_5th_last_half.index) 
    else:
        return pd.DataFrame(ttm_change_6, columns=['{} ttmChange'.format(df_object)], index=ttm_df_6th_last_half.index)   
    


def vectorized_beta(dependents, independent, allowed_missing, out=None):
    """
    Compute slopes of linear regressions between columns of ``dependents`` and
    ``independent``.
    Parameters
    ----------
    dependents : np.array[N, M]
        Array with columns of data to be regressed against ``independent``.
    independent : np.array[N, 1]
        Independent variable of the regression
    allowed_missing : int
        Number of allowed missing (NaN) observations per column. Columns with
        more than this many non-nan observations in both ``dependents`` and
        ``independents`` will output NaN as the regression coefficient.
    Returns
    -------
    slopes : np.array[M]
        Linear regression coefficients for each column of ``dependents``.
    """
    # Cache these as locals since we're going to call them multiple times.
    nan = np.nan
    isnan = np.isnan
    nanmean = np.nanmean
    N, M = dependents.shape

    if out is None:
        out = np.full(M, nan)

    # Copy N times as a column vector and fill with nans to have the same
    # missing value pattern as the dependent variable.
    
    

    # shape: (N, M)
    independent = np.where(
        isnan(dependents),
        nan,
        independent,
    )

    # Calculate beta as Cov(X, Y) / Cov(X, X).
    # 
    #
    # NOTE: The usual formula for covariance is::
    #
    #    mean((X - mean(X)) * (Y - mean(Y)))
    #
    # However, we don't actually need to take the mean of both sides of the
    # product, because of the following equivalence::
    #
    # Let X_res = (X - mean(X)).
    # We have:
    #
    #     mean(X_res * (Y - mean(Y))) = mean(X_res * (Y - mean(Y)))
    #                             (1) = mean((X_res * Y) - (X_res * mean(Y)))
    #                             (2) = mean(X_res * Y) - mean(X_res * mean(Y))
    #                             (3) = mean(X_res * Y) - mean(X_res) * mean(Y)
    #                             (4) = mean(X_res * Y) - 0 * mean(Y)
    #                             (5) = mean(X_res * Y)
    #
    #
    # The tricky step in the above derivation is step (4). We know that
    # mean(X_res) is zero because, for any X:
    #
    #     mean(X - mean(X)) = mean(X) - mean(X) = 0.
    #
    # The upshot of this is that we only have to center one of `independent`
    # and `dependent` when calculating covariances. Since we need the centered
    # `independent` to calculate its variance in the next step, we choose to
    # center `independent`.

    # shape: (N, M)
    ind_residual = independent - nanmean(independent, axis=0)

    # shape: (M,)
    covariances = nanmean(ind_residual * dependents, axis=0)

    # We end up with different variances in each column here because each
    # column may have a different subset of the data dropped due to missing
    # data in the corresponding dependent column.
    # shape: (M,)
    independent_variances = nanmean(ind_residual ** 2, axis=0)

    # shape: (M,)
    np.divide(covariances, independent_variances, out=out)

    # Write nans back to locations where we have more then allowed number of
    # missing entries.
    nanlocs = isnan(independent).sum(axis=0) > allowed_missing
    out[nanlocs] = nan

    return out

def compute_beta(benchmark_rets_df, peer_group_rets_df, allowed_missing_percentage):
    
    returns = peer_group_rets_df.values
    benchmark_returns = benchmark_rets_df.values
    allowed_missing_count = int(allowed_missing_percentage*len(returns))
    beta = vectorized_beta(dependents=returns, independent=benchmark_returns, allowed_missing=allowed_missing_count)
    
    return pd.DataFrame(beta, index=peer_group_rets_df.columns, columns=['beta'])



def re_concat(df1, df2, df3, df4, df5, df6):
    """
    df1: most recent df.
    df6: less recent df.
    """
    new_df = pd.concat([
        df6.rename(columns={df6.columns.values.item():'2017 Q4'}),
        df5.rename(columns={df5.columns.values.item():'2018 Q2'}),
        df4.rename(columns={df4.columns.values.item():'2018 Q4'}),
        df3.rename(columns={df3.columns.values.item():'2019 Q2'}),
        df2.rename(columns={df2.columns.values.item():'2019 Q4'}),
        df1.rename(columns={df1.columns.values.item():'2020 Q2'}),
    ], axis=1).dropna()
     
    return new_df

def check_plot(df, df_plot_item):
    
    fig = plt.figure(figsize=(8,5))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.236, 2])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    sns.boxplot(data=df, width=0.25, palette='Blues', linewidth=1.4, fliersize=1, showmeans=True, showfliers=False, ax=ax0)
    for patch in ax0.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .4))
    sns.lineplot(data=pd.DataFrame(df.apply(lambda x: x.median())), sort=False, markers='s', markersize=8, linewidth=3, palette='Blues', legend=False, ax=ax1)
    ax0.margins(y=0.06)
    ax0.set_title(df_plot_item + ' boxplot')
    ax1.set_title(df_plot_item + ' - median')
    plt.tight_layout()

def check_trans_dist(df_original, df_transformed):
    
    for col in df_original.columns:
        f, axes = plt.subplots(1, 2, figsize=(10, 3))
        sns.histplot(df_original[col], color='seagreen', ax=axes[0], kde=True)
        axes[0].text(x=0.99, y=0.98, s="Skewness: %f" % df_original[col].skew(), transform=axes[0].transAxes, fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right', backgroundcolor='white')
        axes[0].text(x=0.99, y=0.9, s="Kurtosis: %f" % df_original[col].kurt(), transform=axes[0].transAxes, fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right', backgroundcolor='white')
        axes[0].set_title('original data')
        sns.histplot(df_transformed[col], color='deepskyblue', ax=axes[1], kde=True)
        axes[1].text(x=0.99, y=0.98, s="Skewness: %f" % df_transformed[col].skew(), transform=axes[1].transAxes, fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right', backgroundcolor='white')
        axes[1].text(x=0.99, y=0.9, s="Kurtosis: %f" % df_transformed[col].kurt(), transform=axes[1].transAxes, fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right', backgroundcolor='white')
        axes[1].set_title('data after transformation')
        plt.tight_layout()
        #plt.savefig(fname=('transformed_{}'.format(col)).replace(' ','_'), dpi=100)

        
    

ridgeCV_model = RidgeCV(cv=5)
lassoCV_model = LassoCV(cv=5)
elnetCV_model = ElasticNetCV(cv=5)

def fit_my_models(independent_df, dependent_df):
    
    
    independent = independent_df.values
    dependent = dependent_df.values
    
    ridge_coefs = []
    lasso_coefs = []
    elnet_coefs = []
    ridge_pred = []
    lasso_pred = []
    elnet_pred = []
    
    
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(independent):
        
        X_train, X_test = independent[train_index], independent[test_index]
        y_train, y_test = dependent[train_index], dependent[test_index]
        
        clf_ridge = ridgeCV_model.fit(X_train, y_train.ravel())
        clf_lasso = lassoCV_model.fit(X_train, y_train.ravel())
        clf_elnet = elnetCV_model.fit(X_train, y_train.ravel())
        
        ridge_coefs.append(clf_ridge.coef_)
        lasso_coefs.append(clf_lasso.coef_)
        elnet_coefs.append(clf_elnet.coef_)
        ridge_pred.append(np.exp(clf_ridge.predict(X_test)))
        lasso_pred.append(np.exp(clf_lasso.predict(X_test)))
        elnet_pred.append(np.exp(clf_elnet.predict(X_test)))
        
    ridge_coefs_df = pd.DataFrame(ridge_coefs, columns=independent_df.columns, index=independent_df.index)
    lasso_coefs_df = pd.DataFrame(lasso_coefs, columns=independent_df.columns, index=independent_df.index)
    elnet_coefs_df = pd.DataFrame(elnet_coefs, columns=independent_df.columns, index=independent_df.index)
    
    return ridge_coefs_df, lasso_coefs_df, elnet_coefs_df, np.array(ridge_pred), np.array(lasso_pred), np.array(elnet_pred)
    



def interim_results_df(start_df, actual_multiple_name, pred_multiples_array, fundamental_value_driver_name):
    
    new_df = pd.DataFrame(start_df['{}'.format(actual_multiple_name)].values, columns=['true {}'.format(actual_multiple_name)], index=start_df.index)
    new_df['estimated {}'.format(actual_multiple_name)] = pred_multiples_array
    new_df['{}'.format(fundamental_value_driver_name)] = start_df['{}'.format(fundamental_value_driver_name)]
    new_df['actual market price'] = start_df['closePriceAdj']
    new_df['estimated price'] = (new_df['estimated {}'.format(actual_multiple_name)]*new_df['{}'.format(fundamental_value_driver_name)])
    new_df['valuation error'] = (new_df['estimated price'] - new_df['actual market price']) / new_df['actual market price']
    new_df['absolute valuation error'] = new_df['valuation error'].abs()
    
    return new_df

def interim_accuracy_df(results_df, index_name):
    
    total_peers_num = len(results_df)
    
    within_10_perc_val_num = len(results_df[results_df['absolute valuation error'] < 0.10])
    within_25_perc_val_num = len(results_df[results_df['absolute valuation error'] < 0.25])
    within_50_perc_val_num = len(results_df[results_df['absolute valuation error'] < 0.50])
    within_75_perc_val_num = len(results_df[results_df['absolute valuation error'] < 0.75])
    within_90_perc_val_num = len(results_df[results_df['absolute valuation error'] < 0.90])
    
    within_10_perc_val_perc = (within_10_perc_val_num / total_peers_num)*100
    within_25_perc_val_perc = (within_25_perc_val_num / total_peers_num)*100
    within_50_perc_val_perc = (within_50_perc_val_num / total_peers_num)*100
    within_75_perc_val_perc = (within_75_perc_val_num / total_peers_num)*100
    within_90_perc_val_perc = (within_90_perc_val_num / total_peers_num)*100
    
    accuracy_dict = {
        '% of valuations within 10% of actual price':[round(within_10_perc_val_perc, 2)],
        '% of valuations within 25% of actual price':[round(within_25_perc_val_perc, 2)],
        '% of valuations within 50% of actual price':[round(within_50_perc_val_perc, 2)],
        '% of valuations within 75% of actual price':[round(within_75_perc_val_perc, 2)],
        '% of valuations within 90% of actual price':[round(within_90_perc_val_perc, 2)],
    }
    
    accuracy_df = pd.DataFrame(accuracy_dict, index=[index_name])
    
    return accuracy_df




def accuracy_overtime_df(acc_df1, acc_df2, acc_df3, acc_df4, acc_df5, acc_df6, accuracy_percentage):
    
    df = pd.DataFrame(pd.concat([
        acc_df1['% of valuations within {}% of actual price'.format(accuracy_percentage)],
        acc_df2['% of valuations within {}% of actual price'.format(accuracy_percentage)],
        acc_df3['% of valuations within {}% of actual price'.format(accuracy_percentage)],
        acc_df4['% of valuations within {}% of actual price'.format(accuracy_percentage)],
        acc_df5['% of valuations within {}% of actual price'.format(accuracy_percentage)],
        acc_df6['% of valuations within {}% of actual price'.format(accuracy_percentage)],
    ])[::-1])
    
    df['date'] = [valuation_date6, valuation_date5, valuation_date4, valuation_date3, valuation_date2, valuation_date1]
    
    return df

def overtime_accuracy_all_df(df_acc_ridge, df_acc_lasso, df_acc_elnet, m_name):

    
    new_df = pd.DataFrame(pd.concat([df_acc_ridge, df_acc_lasso, df_acc_elnet]))
    ridge_s = pd.Series(['ridge {}'.format(m_name)]*len(df_acc_ridge))
    lasso_s = pd.Series(['lasso {}'.format(m_name)]*len(df_acc_lasso))
    elnet_s = pd.Series(['elnet {}'.format(m_name)]*len(df_acc_elnet))
    m_names = np.array(pd.concat([ridge_s, lasso_s, elnet_s]))
    new_df['valuation multiple'] = m_names
    
    return new_df

def overall_accuracy(df1, df2, df3, df4, df5, df6, index_name):
    
    valuation_errors_list = [
        df1['absolute valuation error'],
        df2['absolute valuation error'],
        df3['absolute valuation error'],
        df4['absolute valuation error'],
        df5['absolute valuation error'],
        df6['absolute valuation error'],
    ]
    
    total_valuation_errors = pd.DataFrame(pd.concat(valuation_errors_list))
    tot_peers_num = len(total_valuation_errors)
    
    within_10_perc_val_num = len(total_valuation_errors[total_valuation_errors['absolute valuation error'] < 0.10])
    within_25_perc_val_num = len(total_valuation_errors[total_valuation_errors['absolute valuation error'] < 0.25])
    within_50_perc_val_num = len(total_valuation_errors[total_valuation_errors['absolute valuation error'] < 0.50])
    within_75_perc_val_num = len(total_valuation_errors[total_valuation_errors['absolute valuation error'] < 0.75])
    within_90_perc_val_num = len(total_valuation_errors[total_valuation_errors['absolute valuation error'] < 0.90])
    
    within_10_perc_val_perc = (within_10_perc_val_num / tot_peers_num)*100
    within_25_perc_val_perc = (within_25_perc_val_num / tot_peers_num)*100
    within_50_perc_val_perc = (within_50_perc_val_num / tot_peers_num)*100
    within_75_perc_val_perc = (within_75_perc_val_num / tot_peers_num)*100
    within_90_perc_val_perc = (within_90_perc_val_num / tot_peers_num)*100
    
    accuracy_dict = {
        '% of valuations within 10% of actual price':[round(within_10_perc_val_perc, 2)],
        '% of valuations within 25% of actual price':[round(within_25_perc_val_perc, 2)],
        '% of valuations within 50% of actual price':[round(within_50_perc_val_perc, 2)],
        '% of valuations within 75% of actual price':[round(within_75_perc_val_perc, 2)],
        '% of valuations within 90% of actual price':[round(within_90_perc_val_perc, 2)],
    }
    
    tot_acc_df = pd.DataFrame(accuracy_dict, index=[index_name])
    
    return tot_acc_df


def overtime_accuracy_plot(overtime_accuracy_df, accuracy_percentage, gen_multiple_name):
    
    fig = plt.figure(figsize=(9,4))
    sns.lineplot(data=overtime_accuracy_df, x='date', y='% of valuations within {}% of actual price'.format(accuracy_percentage), style='valuation multiple', hue='valuation multiple', sort=False, markers=['s','s','s'], dashes=False, palette='viridis', linewidth=1.8)
    plt.margins(y=0.25)
    plt.title('{} accuracy over time'.format(gen_multiple_name), fontweight='bold')
    plt.tight_layout()
    #plt.savefig(fname=('{0}_{1}_overtime_accuracy'.format(gen_multiple_name, accuracy_percentage)).replace('/','_'), dpi=100)
    plt.close()
    
    return fig



def ve_descriptive_statistics(df1, df2, df3, df4, df5, df6, estimator_name, multiple_name):

    ve1 = df1['valuation error']
    ve2 = df2['valuation error']
    ve3 = df3['valuation error']
    ve4 = df4['valuation error']
    ve5 = df5['valuation error']
    ve6 = df6['valuation error']
    
    overall_valuation_error = pd.concat([ve1, ve2, ve3, ve4, ve5, ve6])
    overall_absolute_ve = pd.concat([ve1.abs(), ve2.abs(), ve3.abs(), ve4.abs(), ve5.abs(), ve6.abs()])

    ve_mse = (overall_valuation_error**2).mean()
    ve_rmse = np.sqrt(ve_mse)
    ve_mae = overall_absolute_ve.mean()
    
    err_dict = {'RMSE':round(ve_rmse, 3),
                'MAE':round(ve_mae, 3)}
    
    new_df = pd.DataFrame(err_dict, index=['{0} {1}'.format(estimator_name, multiple_name)])

    return new_df


def overtime_ve_desc_stat(df1, df2, df3, df4, df5, df6, estimator_name, multiple_name):

    ve1 = df1['valuation error']
    ve2 = df2['valuation error']
    ve3 = df3['valuation error']
    ve4 = df4['valuation error']
    ve5 = df5['valuation error']
    ve6 = df6['valuation error']

    ve1_mse = (ve1**2).mean()
    ve2_mse = (ve2**2).mean()
    ve3_mse = (ve3**2).mean()
    ve4_mse = (ve4**2).mean()
    ve5_mse = (ve5**2).mean()
    ve6_mse = (ve6**2).mean()

    ve1_rmse = np.sqrt(ve1_mse)
    ve2_rmse = np.sqrt(ve2_mse)
    ve3_rmse = np.sqrt(ve3_mse)
    ve4_rmse = np.sqrt(ve4_mse)
    ve5_rmse = np.sqrt(ve5_mse)
    ve6_rmse = np.sqrt(ve6_mse)

    ve1_mae = ve1.abs().mean()
    ve2_mae = ve2.abs().mean()
    ve3_mae = ve3.abs().mean()
    ve4_mae = ve4.abs().mean()
    ve5_mae = ve5.abs().mean()
    ve6_mae = ve6.abs().mean()

    err_dict_rmse = {
        '2020-10-01': round(ve1_rmse, 3),
        '2020-04-01': round(ve2_rmse, 3), 
        '2019-10-01': round(ve3_rmse, 3),
        '2019-04-01': round(ve4_rmse, 3), 
        '2018-10-01': round(ve5_rmse, 3),
        '2018-04-02': round(ve6_rmse, 3),
    }

    err_dict_mae = {
        '2020-10-01': round(ve1_mae, 3),
        '2020-04-01': round(ve2_mae, 3), 
        '2019-10-01': round(ve3_mae, 3),
        '2019-04-01': round(ve4_mae, 3), 
        '2018-10-01': round(ve5_mae, 3),
        '2018-04-02': round(ve6_mae, 3),
    }

    

    rmse_new_df = pd.DataFrame(err_dict_rmse, index=['{0} {1}'.format(estimator_name, multiple_name)]).transpose()   
    

    mae_new_df = pd.DataFrame(err_dict_mae, index=['{0} {1}'.format(estimator_name, multiple_name)]).transpose()
    

    return rmse_new_df, mae_new_df



def check_predictions_results(
    
    multiple_name,
    ridge_res_1,
    ridge_res_2,
    ridge_res_3,
    ridge_res_4,
    ridge_res_5,
    ridge_res_6,
    lasso_res_1,
    lasso_res_2,
    lasso_res_3,
    lasso_res_4,
    lasso_res_5,
    lasso_res_6,
    elnet_res_1,
    elnet_res_2,
    elnet_res_3,
    elnet_res_4,
    elnet_res_5,
    elnet_res_6,
    
):
    
    # accuracy for each half 
    
    ridge_acc_df1 = interim_accuracy_df(ridge_res_1, 'ridge {0} {1}'.format(multiple_name, valuation_date1))
    ridge_acc_df2 = interim_accuracy_df(ridge_res_2, 'ridge {0} {1}'.format(multiple_name, valuation_date2))
    ridge_acc_df3 = interim_accuracy_df(ridge_res_3, 'ridge {0} {1}'.format(multiple_name, valuation_date3))
    ridge_acc_df4 = interim_accuracy_df(ridge_res_4, 'ridge {0} {1}'.format(multiple_name, valuation_date4))
    ridge_acc_df5 = interim_accuracy_df(ridge_res_5, 'ridge {0} {1}'.format(multiple_name, valuation_date5))
    ridge_acc_df6 = interim_accuracy_df(ridge_res_6, 'ridge {0} {1}'.format(multiple_name, valuation_date6))
    
    lasso_acc_df1 = interim_accuracy_df(lasso_res_1, 'lasso {0} {1}'.format(multiple_name, valuation_date1))
    lasso_acc_df2 = interim_accuracy_df(lasso_res_2, 'lasso {0} {1}'.format(multiple_name, valuation_date2))
    lasso_acc_df3 = interim_accuracy_df(lasso_res_3, 'lasso {0} {1}'.format(multiple_name, valuation_date3))
    lasso_acc_df4 = interim_accuracy_df(lasso_res_4, 'lasso {0} {1}'.format(multiple_name, valuation_date4))
    lasso_acc_df5 = interim_accuracy_df(lasso_res_5, 'lasso {0} {1}'.format(multiple_name, valuation_date5))
    lasso_acc_df6 = interim_accuracy_df(lasso_res_6, 'lasso {0} {1}'.format(multiple_name, valuation_date6))
    
    elnet_acc_df1 = interim_accuracy_df(elnet_res_1, 'elnet {0} {1}'.format(multiple_name, valuation_date1))
    elnet_acc_df2 = interim_accuracy_df(elnet_res_2, 'elnet {0} {1}'.format(multiple_name, valuation_date2))
    elnet_acc_df3 = interim_accuracy_df(elnet_res_3, 'elnet {0} {1}'.format(multiple_name, valuation_date3))
    elnet_acc_df4 = interim_accuracy_df(elnet_res_4, 'elnet {0} {1}'.format(multiple_name, valuation_date4))
    elnet_acc_df5 = interim_accuracy_df(elnet_res_5, 'elnet {0} {1}'.format(multiple_name, valuation_date5))
    elnet_acc_df6 = interim_accuracy_df(elnet_res_6, 'elnet {0} {1}'.format(multiple_name, valuation_date6))
    
    # acc. grouped by estimator 
    acc_df1 = pd.concat([ridge_acc_df1, lasso_acc_df1, elnet_acc_df1])
    acc_df2 = pd.concat([ridge_acc_df2, lasso_acc_df2, elnet_acc_df2])
    acc_df3 = pd.concat([ridge_acc_df3, lasso_acc_df3, elnet_acc_df3])
    acc_df4 = pd.concat([ridge_acc_df4, lasso_acc_df4, elnet_acc_df4])
    acc_df5 = pd.concat([ridge_acc_df5, lasso_acc_df5, elnet_acc_df5])
    acc_df6 = pd.concat([ridge_acc_df6, lasso_acc_df6, elnet_acc_df6])
    
    overtime_accuracy_df = pd.concat([acc_df1, acc_df2, acc_df3, acc_df4, acc_df5, acc_df6])
    
    
    # overtime accuracy grouped by 25% accuracy for each estimator 
    ridge_25_acc = accuracy_overtime_df(ridge_acc_df1, ridge_acc_df2, ridge_acc_df3, ridge_acc_df4, ridge_acc_df5, ridge_acc_df6, '25')
    lasso_25_acc = accuracy_overtime_df(lasso_acc_df1, lasso_acc_df2, lasso_acc_df3, lasso_acc_df4, lasso_acc_df5, lasso_acc_df6, '25')
    elnet_25_acc = accuracy_overtime_df(elnet_acc_df1, elnet_acc_df2, elnet_acc_df3, elnet_acc_df4, elnet_acc_df5, elnet_acc_df6, '25')
    
    m_25_acc = overtime_accuracy_all_df(ridge_25_acc, lasso_25_acc, elnet_25_acc, multiple_name)
    
    # 25% accuracy plot 
    acc_plot_25 = overtime_accuracy_plot(m_25_acc, '25', multiple_name)
    
    # overall accuracy 
    ridge_overall_acc = overall_accuracy(ridge_res_1, ridge_res_2, ridge_res_3, ridge_res_4, ridge_res_5, ridge_res_6, 'ridge {}'.format(multiple_name))
    lasso_overall_acc = overall_accuracy(lasso_res_1, lasso_res_2, lasso_res_3, lasso_res_4, lasso_res_5, lasso_res_6, 'lasso {}'.format(multiple_name))
    elnet_overall_acc = overall_accuracy(elnet_res_1, elnet_res_2, elnet_res_3, elnet_res_4, elnet_res_5, elnet_res_6, 'elnet {}'.format(multiple_name))
    overall_accuracy_df = pd.concat([ridge_overall_acc, lasso_overall_acc, elnet_overall_acc])

    # valuation error descriptive statistics (RMSE, MAE)
    ridge_ds = ve_descriptive_statistics(ridge_res_1, ridge_res_2, ridge_res_3, ridge_res_4, ridge_res_5, ridge_res_6, 'ridge', multiple_name)
    lasso_ds = ve_descriptive_statistics(lasso_res_1, lasso_res_2, lasso_res_3, lasso_res_4, lasso_res_5, lasso_res_6, 'lasso', multiple_name)
    elnet_ds = ve_descriptive_statistics(elnet_res_1, elnet_res_2, elnet_res_3, elnet_res_4, elnet_res_5, elnet_res_6, 'elnet', multiple_name)   
    ve_desc_stat_df = pd.concat([ridge_ds, lasso_ds, elnet_ds])

    # overtime valuation errors 
    ridge_ot_rmse, ridge_ot_mae = overtime_ve_desc_stat(ridge_res_1, ridge_res_2, ridge_res_3, ridge_res_4, ridge_res_5, ridge_res_6, 'ridge', multiple_name)
    lasso_ot_rmse, lasso_ot_mae = overtime_ve_desc_stat(lasso_res_1, lasso_res_2, lasso_res_3, lasso_res_4, lasso_res_5, lasso_res_6, 'lasso', multiple_name)
    elnet_ot_rmse, elnet_ot_mae = overtime_ve_desc_stat(elnet_res_1, elnet_res_2, elnet_res_3, elnet_res_4, elnet_res_5, elnet_res_6, 'elnet', multiple_name)

    ot_ve_rmse = pd.concat([ridge_ot_rmse, lasso_ot_rmse, elnet_ot_rmse], axis=1)
    ot_ve_mae = pd.concat([ridge_ot_mae, lasso_ot_mae, elnet_ot_mae], axis=1)

    #ot_ve_dates = pd.Series(data=['2020-10-01', '2020-04-01', '2019-10-01', '2019-04-01', '2018-10-01', '2018-04-02'])
    #ot_ve_rmse['date'] = ot_ve_dates
    #ot_ve_mae['date'] = ot_ve_dates

    return overtime_accuracy_df, overall_accuracy_df, acc_plot_25, ve_desc_stat_df, ot_ve_rmse, ot_ve_mae




def plotting_dist_coefs(multiple_df, coefs_df, multiple_name, estimator_name):
    
    fig = plt.figure(figsize=(8,9))
    gs = gridspec.GridSpec(3,1, height_ratios=[1, 1.5, 0.05])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    sns.boxplot(x='date', y=multiple_name, data=multiple_df, hue=' ', width=0.5, linewidth=1.4, showfliers=False, hue_order=['true', 'estimated'], palette='bwr', ax=ax0)
    sns.heatmap(data=coefs_df, linewidth=3, cmap='bwr', center=0, ax=ax1, cbar=True, cbar_ax=ax2, cbar_kws={"orientation": "horizontal"})
    ax0.set_title('{0} {1} - boxplot & median value driver coefs over time'.format(estimator_name, multiple_name), fontweight='bold')
    plt.tight_layout()
    #plt.savefig(fname=('{0}_{1}_overtime_coefs'.format(estimator_name, multiple_name)).replace('/','_'), dpi=100)
    plt.close()
    
    
    return fig


def check_drivers_impact(
    
    multiple_name,
    
    ridge_res1,
    ridge_res2,
    ridge_res3,
    ridge_res4,
    ridge_res5,
    ridge_res6,
    lasso_res1,
    lasso_res2,
    lasso_res3,
    lasso_res4,
    lasso_res5,
    lasso_res6,
    elnet_res1,
    elnet_res2,
    elnet_res3,
    elnet_res4,
    elnet_res5,
    elnet_res6,
    
    ridge_coefs1, 
    ridge_coefs2,
    ridge_coefs3,
    ridge_coefs4,
    ridge_coefs5,
    ridge_coefs6,
    lasso_coefs1,
    lasso_coefs2,
    lasso_coefs3,
    lasso_coefs4,
    lasso_coefs5,
    lasso_coefs6,
    elnet_coefs1,
    elnet_coefs2,
    elnet_coefs3,
    elnet_coefs4,
    elnet_coefs5,
    elnet_coefs6,
        
):
    
    ridge_true_m1 = pd.DataFrame(ridge_res1['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_true_m2 = pd.DataFrame(ridge_res2['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_true_m3 = pd.DataFrame(ridge_res3['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_true_m4 = pd.DataFrame(ridge_res4['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_true_m5 = pd.DataFrame(ridge_res5['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_true_m6 = pd.DataFrame(ridge_res6['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    
    lasso_true_m1 = pd.DataFrame(lasso_res1['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_true_m2 = pd.DataFrame(lasso_res2['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_true_m3 = pd.DataFrame(lasso_res3['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_true_m4 = pd.DataFrame(lasso_res4['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_true_m5 = pd.DataFrame(lasso_res5['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_true_m6 = pd.DataFrame(lasso_res6['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    
    elnet_true_m1 = pd.DataFrame(elnet_res1['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_true_m2 = pd.DataFrame(elnet_res2['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_true_m3 = pd.DataFrame(elnet_res3['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_true_m4 = pd.DataFrame(elnet_res4['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_true_m5 = pd.DataFrame(elnet_res5['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_true_m6 = pd.DataFrame(elnet_res6['true {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    
    ridge_est_m1 = pd.DataFrame(ridge_res1['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_est_m2 = pd.DataFrame(ridge_res2['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_est_m3 = pd.DataFrame(ridge_res3['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_est_m4 = pd.DataFrame(ridge_res4['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_est_m5 = pd.DataFrame(ridge_res5['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    ridge_est_m6 = pd.DataFrame(ridge_res6['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    
    lasso_est_m1 = pd.DataFrame(lasso_res1['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_est_m2 = pd.DataFrame(lasso_res2['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_est_m3 = pd.DataFrame(lasso_res3['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_est_m4 = pd.DataFrame(lasso_res4['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_est_m5 = pd.DataFrame(lasso_res5['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    lasso_est_m6 = pd.DataFrame(lasso_res6['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    
    elnet_est_m1 = pd.DataFrame(elnet_res1['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_est_m2 = pd.DataFrame(elnet_res2['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_est_m3 = pd.DataFrame(elnet_res3['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_est_m4 = pd.DataFrame(elnet_res4['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_est_m5 = pd.DataFrame(elnet_res5['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    elnet_est_m6 = pd.DataFrame(elnet_res6['estimated {}'.format(multiple_name)].rename('{}'.format(multiple_name)))
    
    ridge_true_m1[' '] = pd.Series(['true']*len(ridge_true_m1)).values
    ridge_true_m2[' '] = pd.Series(['true']*len(ridge_true_m2)).values
    ridge_true_m3[' '] = pd.Series(['true']*len(ridge_true_m3)).values
    ridge_true_m4[' '] = pd.Series(['true']*len(ridge_true_m4)).values
    ridge_true_m5[' '] = pd.Series(['true']*len(ridge_true_m5)).values
    ridge_true_m6[' '] = pd.Series(['true']*len(ridge_true_m6)).values
    
    lasso_true_m1[' '] = pd.Series(['true']*len(lasso_true_m1)).values
    lasso_true_m2[' '] = pd.Series(['true']*len(lasso_true_m2)).values
    lasso_true_m3[' '] = pd.Series(['true']*len(lasso_true_m3)).values
    lasso_true_m4[' '] = pd.Series(['true']*len(lasso_true_m4)).values
    lasso_true_m5[' '] = pd.Series(['true']*len(lasso_true_m5)).values
    lasso_true_m6[' '] = pd.Series(['true']*len(lasso_true_m6)).values
    
    elnet_true_m1[' '] = pd.Series(['true']*len(elnet_true_m1)).values
    elnet_true_m2[' '] = pd.Series(['true']*len(elnet_true_m2)).values
    elnet_true_m3[' '] = pd.Series(['true']*len(elnet_true_m3)).values
    elnet_true_m4[' '] = pd.Series(['true']*len(elnet_true_m4)).values
    elnet_true_m5[' '] = pd.Series(['true']*len(elnet_true_m5)).values
    elnet_true_m6[' '] = pd.Series(['true']*len(elnet_true_m6)).values
    
    ridge_est_m1[' '] = pd.Series(['estimated']*len(ridge_est_m1)).values
    ridge_est_m2[' '] = pd.Series(['estimated']*len(ridge_est_m2)).values
    ridge_est_m3[' '] = pd.Series(['estimated']*len(ridge_est_m3)).values
    ridge_est_m4[' '] = pd.Series(['estimated']*len(ridge_est_m4)).values
    ridge_est_m5[' '] = pd.Series(['estimated']*len(ridge_est_m5)).values
    ridge_est_m6[' '] = pd.Series(['estimated']*len(ridge_est_m6)).values
    
    lasso_est_m1[' '] = pd.Series(['estimated']*len(lasso_est_m1)).values
    lasso_est_m2[' '] = pd.Series(['estimated']*len(lasso_est_m2)).values
    lasso_est_m3[' '] = pd.Series(['estimated']*len(lasso_est_m3)).values
    lasso_est_m4[' '] = pd.Series(['estimated']*len(lasso_est_m4)).values
    lasso_est_m5[' '] = pd.Series(['estimated']*len(lasso_est_m5)).values
    lasso_est_m6[' '] = pd.Series(['estimated']*len(lasso_est_m6)).values
    
    elnet_est_m1[' '] = pd.Series(['estimated']*len(elnet_est_m1)).values
    elnet_est_m2[' '] = pd.Series(['estimated']*len(elnet_est_m2)).values
    elnet_est_m3[' '] = pd.Series(['estimated']*len(elnet_est_m3)).values
    elnet_est_m4[' '] = pd.Series(['estimated']*len(elnet_est_m4)).values
    elnet_est_m5[' '] = pd.Series(['estimated']*len(elnet_est_m5)).values
    elnet_est_m6[' '] = pd.Series(['estimated']*len(elnet_est_m6)).values
    
    
    ridge_m1 = pd.concat([ridge_true_m1, ridge_est_m1])
    ridge_m2 = pd.concat([ridge_true_m2, ridge_est_m2])
    ridge_m3 = pd.concat([ridge_true_m3, ridge_est_m3])
    ridge_m4 = pd.concat([ridge_true_m4, ridge_est_m4])
    ridge_m5 = pd.concat([ridge_true_m5, ridge_est_m5])
    ridge_m6 = pd.concat([ridge_true_m6, ridge_est_m6])
    
    lasso_m1 = pd.concat([lasso_true_m1, lasso_est_m1])
    lasso_m2 = pd.concat([lasso_true_m2, lasso_est_m2])
    lasso_m3 = pd.concat([lasso_true_m3, lasso_est_m3])
    lasso_m4 = pd.concat([lasso_true_m4, lasso_est_m4])
    lasso_m5 = pd.concat([lasso_true_m5, lasso_est_m5])
    lasso_m6 = pd.concat([lasso_true_m6, lasso_est_m6])
    
    elnet_m1 = pd.concat([elnet_true_m1, elnet_est_m1])
    elnet_m2 = pd.concat([elnet_true_m2, elnet_est_m2])
    elnet_m3 = pd.concat([elnet_true_m3, elnet_est_m3])
    elnet_m4 = pd.concat([elnet_true_m4, elnet_est_m4])
    elnet_m5 = pd.concat([elnet_true_m5, elnet_est_m5])
    elnet_m6 = pd.concat([elnet_true_m6, elnet_est_m6])
    
    ridge_m1['date'] = pd.Series([valuation_date1]*len(ridge_m1)).values
    ridge_m2['date'] = pd.Series([valuation_date2]*len(ridge_m2)).values
    ridge_m3['date'] = pd.Series([valuation_date3]*len(ridge_m3)).values
    ridge_m4['date'] = pd.Series([valuation_date4]*len(ridge_m4)).values
    ridge_m5['date'] = pd.Series([valuation_date5]*len(ridge_m5)).values
    ridge_m6['date'] = pd.Series([valuation_date6]*len(ridge_m6)).values
    
    lasso_m1['date'] = pd.Series([valuation_date1]*len(lasso_m1)).values
    lasso_m2['date'] = pd.Series([valuation_date2]*len(lasso_m2)).values
    lasso_m3['date'] = pd.Series([valuation_date3]*len(lasso_m3)).values
    lasso_m4['date'] = pd.Series([valuation_date4]*len(lasso_m4)).values
    lasso_m5['date'] = pd.Series([valuation_date5]*len(lasso_m5)).values
    lasso_m6['date'] = pd.Series([valuation_date6]*len(lasso_m6)).values
    
    elnet_m1['date'] = pd.Series([valuation_date1]*len(elnet_m1)).values
    elnet_m2['date'] = pd.Series([valuation_date2]*len(elnet_m2)).values
    elnet_m3['date'] = pd.Series([valuation_date3]*len(elnet_m3)).values
    elnet_m4['date'] = pd.Series([valuation_date4]*len(elnet_m4)).values
    elnet_m5['date'] = pd.Series([valuation_date5]*len(elnet_m5)).values
    elnet_m6['date'] = pd.Series([valuation_date6]*len(elnet_m6)).values
    
    
    ridge_m = pd.concat([
        ridge_m1,
        ridge_m2,
        ridge_m3,
        ridge_m4,
        ridge_m5,
        ridge_m6,
    ])[::-1]
    
    
    lasso_m = pd.concat([
        lasso_m1,
        lasso_m2,
        lasso_m3,
        lasso_m4,
        lasso_m5,
        lasso_m6,    
    ])[::-1]
    
    
    elnet_m = pd.concat([
        elnet_m1,
        elnet_m2,
        elnet_m3,
        elnet_m4,
        elnet_m5,
        elnet_m6,    
    ])[::-1]
    
    
    
    
    ridge_c1 = pd.DataFrame(ridge_coefs1.apply(lambda x: x.median()).rename(valuation_date1))
    ridge_c2 = pd.DataFrame(ridge_coefs2.apply(lambda x: x.median()).rename(valuation_date2))
    ridge_c3 = pd.DataFrame(ridge_coefs3.apply(lambda x: x.median()).rename(valuation_date3))
    ridge_c4 = pd.DataFrame(ridge_coefs4.apply(lambda x: x.median()).rename(valuation_date4))
    ridge_c5 = pd.DataFrame(ridge_coefs5.apply(lambda x: x.median()).rename(valuation_date5))
    ridge_c6 = pd.DataFrame(ridge_coefs6.apply(lambda x: x.median()).rename(valuation_date6))
    
    lasso_c1 = pd.DataFrame(lasso_coefs1.apply(lambda x: x.median()).rename(valuation_date1))
    lasso_c2 = pd.DataFrame(lasso_coefs2.apply(lambda x: x.median()).rename(valuation_date2))
    lasso_c3 = pd.DataFrame(lasso_coefs3.apply(lambda x: x.median()).rename(valuation_date3))
    lasso_c4 = pd.DataFrame(lasso_coefs4.apply(lambda x: x.median()).rename(valuation_date4))
    lasso_c5 = pd.DataFrame(lasso_coefs5.apply(lambda x: x.median()).rename(valuation_date5))
    lasso_c6 = pd.DataFrame(lasso_coefs6.apply(lambda x: x.median()).rename(valuation_date6))
    
    elnet_c1 = pd.DataFrame(elnet_coefs1.apply(lambda x: x.median()).rename(valuation_date1))
    elnet_c2 = pd.DataFrame(elnet_coefs2.apply(lambda x: x.median()).rename(valuation_date2))
    elnet_c3 = pd.DataFrame(elnet_coefs3.apply(lambda x: x.median()).rename(valuation_date3))
    elnet_c4 = pd.DataFrame(elnet_coefs4.apply(lambda x: x.median()).rename(valuation_date4))
    elnet_c5 = pd.DataFrame(elnet_coefs5.apply(lambda x: x.median()).rename(valuation_date5))
    elnet_c6 = pd.DataFrame(elnet_coefs6.apply(lambda x: x.median()).rename(valuation_date6))
    
    ridge_c = pd.concat([ridge_c6, ridge_c5, ridge_c4, ridge_c3, ridge_c2, ridge_c1], axis=1)
    lasso_c = pd.concat([lasso_c6, lasso_c5, lasso_c4, lasso_c3, lasso_c2, lasso_c1], axis=1)
    elnet_c = pd.concat([elnet_c6, elnet_c5, elnet_c4, elnet_c3, elnet_c2, elnet_c1], axis=1)
    
    ridge_fig1 = plotting_dist_coefs(ridge_m, ridge_c, multiple_name, 'ridge')
    lasso_fig1 = plotting_dist_coefs(lasso_m, lasso_c, multiple_name, 'lasso')
    elnet_fig1 = plotting_dist_coefs(elnet_m, elnet_c, multiple_name, 'elnet')
    
    
    # overall coefs
    ridge_overall_coefs_df = pd.concat([ridge_coefs1,
                                        ridge_coefs2,
                                        ridge_coefs3,
                                        ridge_coefs4,
                                        ridge_coefs5,
                                        ridge_coefs6,    
                                       ])
    
    lasso_overall_coefs_df = pd.concat([lasso_coefs1,
                                        lasso_coefs2,
                                        lasso_coefs3,
                                        lasso_coefs4,
                                        lasso_coefs5,
                                        lasso_coefs6,    
                                       ])
    
    elnet_overall_coefs_df = pd.concat([elnet_coefs1,
                                        elnet_coefs2,
                                        elnet_coefs3,
                                        elnet_coefs4,
                                        elnet_coefs5,
                                        elnet_coefs6,    
                                       ])
    
    
    # overall median coefs
    overall_coefs_df = pd.concat([ridge_overall_coefs_df, lasso_overall_coefs_df, elnet_overall_coefs_df])
    
    overall_coefs_fig = plt.figure(figsize=(8,6))
    sns.barplot(data=overall_coefs_df, estimator=np.median, ci=95, orient='h', errwidth=1.5, order=overall_coefs_df.apply(lambda x: x.median()).sort_values(ascending=False).index.tolist(), palette='bwr_r')
    plt.title('{} overall median coefs with 95% confidence interval'.format(multiple_name), fontweight='bold')
    plt.tight_layout()
    #plt.savefig(fname=('{}_overall_with_95'.format(multiple_name)).replace('/','_'), dpi=100)
    plt.close()
    
    # overall median coefs by estimator
    ridge_overall_median_coefs_df = pd.DataFrame(ridge_overall_coefs_df.apply(lambda x: x.median()))
    lasso_overall_median_coefs_df = pd.DataFrame(lasso_overall_coefs_df.apply(lambda x: x.median()))
    elnet_overall_median_coefs_df = pd.DataFrame(elnet_overall_coefs_df.apply(lambda x: x.median()))
    
    overall_median_coefs_by_est = pd.concat([ridge_overall_median_coefs_df,
                                             lasso_overall_median_coefs_df,
                                             elnet_overall_median_coefs_df,
                                            ]).reset_index()
    
    ridge_name_series = pd.Series(['ridge']*len(ridge_overall_median_coefs_df))
    lasso_name_series = pd.Series(['lasso']*len(lasso_overall_median_coefs_df)) 
    elnet_name_series = pd.Series(['elnet']*len(elnet_overall_median_coefs_df)) 
    
    estimator_names = pd.concat([ridge_name_series, lasso_name_series, elnet_name_series]).values
    overall_median_coefs_by_est['estimator'] = estimator_names
    overall_median_coefs_by_est.columns = ['residual value driver', 'median coef', 'estimator']
    
    overall_coefs_by_est_fig = plt.figure(figsize=(10,6))
    sns.barplot(x='residual value driver', y='median coef', data=overall_median_coefs_by_est, hue='estimator', palette='viridis', order=overall_coefs_df.apply(lambda x: x.median()).sort_values(ascending=False).index.tolist())
    plt.xticks(rotation=45, ha='right')
    plt.title('{} overall median coefs by estimator'.format(multiple_name), fontweight='bold')
    plt.tight_layout()
    #plt.savefig(fname=('{}_overall_by_estimator'.format(multiple_name)).replace('/','_'), dpi=100)
    plt.close()
    
    
    
    
    
    return ridge_fig1, lasso_fig1, elnet_fig1, overall_coefs_fig, overall_coefs_by_est_fig





def list_plot_coefs_over_time(multiple_name, df, df25, df75):
    df_t = df.transpose()
    df25_t = df25.transpose()
    df75_t = df75.transpose()
    fig_list = []
    for col in df_t.columns:
        fig = plt.figure(figsize=(6,2))
        sns.lineplot(data=df_t[col].transpose(), linewidth=1.6, color='g')
        plt.axhline(y=0, color='r', ls='--', linewidth=1.2)
        plt.margins(y=1)
        x = df.columns
        plt.fill_between(x, df25_t[col].transpose().values, df75_t[col].transpose().values, color='g', alpha=.2)
        plt.title('{} median coef over time with 25th and 75th percentile'.format(multiple_name))
        plt.tight_layout()
        plt.close()
        fig_list.append(fig)
 
    return fig_list



def coefs_check(
    
    multiple_name,
    ridge_coefs1,
    ridge_coefs2,
    ridge_coefs3,
    ridge_coefs4,
    ridge_coefs5,
    ridge_coefs6,
    lasso_coefs1,
    lasso_coefs2,
    lasso_coefs3,
    lasso_coefs4,
    lasso_coefs5,
    lasso_coefs6,
    elnet_coefs1,
    elnet_coefs2,
    elnet_coefs3,
    elnet_coefs4,
    elnet_coefs5,
    elnet_coefs6,
     
):
    
    
    
    # median coefs over time
    date = [valuation_date6, valuation_date5, valuation_date4, valuation_date3, valuation_date2, valuation_date1]
    
    median_coefs_ot_df1 = pd.concat([ridge_coefs1, lasso_coefs1, elnet_coefs1]).apply(lambda x: x.median())
    median_coefs_ot_df2 = pd.concat([ridge_coefs2, lasso_coefs2, elnet_coefs2]).apply(lambda x: x.median())
    median_coefs_ot_df3 = pd.concat([ridge_coefs3, lasso_coefs3, elnet_coefs3]).apply(lambda x: x.median()) 
    median_coefs_ot_df4 = pd.concat([ridge_coefs4, lasso_coefs4, elnet_coefs4]).apply(lambda x: x.median())
    median_coefs_ot_df5 = pd.concat([ridge_coefs5, lasso_coefs5, elnet_coefs5]).apply(lambda x: x.median())
    median_coefs_ot_df6 = pd.concat([ridge_coefs6, lasso_coefs6, elnet_coefs6]).apply(lambda x: x.median()) 
    median_coefs_ot_df = pd.concat([median_coefs_ot_df1, median_coefs_ot_df2, median_coefs_ot_df3, median_coefs_ot_df4, median_coefs_ot_df5, median_coefs_ot_df6], axis=1)[::-1]
    median_coefs_ot_df.columns = date
    
    perc_75_ot_coefs_df1 = pd.concat([ridge_coefs1, lasso_coefs1, elnet_coefs1]).apply(lambda x: np.percentile(x, 75))
    perc_75_ot_coefs_df2 = pd.concat([ridge_coefs2, lasso_coefs2, elnet_coefs2]).apply(lambda x: np.percentile(x, 75))
    perc_75_ot_coefs_df3 = pd.concat([ridge_coefs3, lasso_coefs3, elnet_coefs3]).apply(lambda x: np.percentile(x, 75))
    perc_75_ot_coefs_df4 = pd.concat([ridge_coefs4, lasso_coefs4, elnet_coefs4]).apply(lambda x: np.percentile(x, 75))
    perc_75_ot_coefs_df5 = pd.concat([ridge_coefs5, lasso_coefs5, elnet_coefs5]).apply(lambda x: np.percentile(x, 75))
    perc_75_ot_coefs_df6 = pd.concat([ridge_coefs6, lasso_coefs6, elnet_coefs6]).apply(lambda x: np.percentile(x, 75))
    perc_75_ot_coefs_df = pd.concat([perc_75_ot_coefs_df1, perc_75_ot_coefs_df2, perc_75_ot_coefs_df3, perc_75_ot_coefs_df4, perc_75_ot_coefs_df5, perc_75_ot_coefs_df6], axis=1)[::-1]
    
    perc_25_ot_coefs_df1 = pd.concat([ridge_coefs1, lasso_coefs1, elnet_coefs1]).apply(lambda x: np.percentile(x, 25))
    perc_25_ot_coefs_df2 = pd.concat([ridge_coefs2, lasso_coefs2, elnet_coefs2]).apply(lambda x: np.percentile(x, 25))
    perc_25_ot_coefs_df3 = pd.concat([ridge_coefs3, lasso_coefs3, elnet_coefs3]).apply(lambda x: np.percentile(x, 25))
    perc_25_ot_coefs_df4 = pd.concat([ridge_coefs4, lasso_coefs4, elnet_coefs4]).apply(lambda x: np.percentile(x, 25))
    perc_25_ot_coefs_df5 = pd.concat([ridge_coefs5, lasso_coefs5, elnet_coefs5]).apply(lambda x: np.percentile(x, 25))
    perc_25_ot_coefs_df6 = pd.concat([ridge_coefs6, lasso_coefs6, elnet_coefs6]).apply(lambda x: np.percentile(x, 25))
    perc_25_ot_coefs_df = pd.concat([perc_25_ot_coefs_df1, perc_25_ot_coefs_df2, perc_25_ot_coefs_df3, perc_25_ot_coefs_df4, perc_25_ot_coefs_df5, perc_25_ot_coefs_df6], axis=1)[::-1]
  
    fig1_list = list_plot_coefs_over_time(multiple_name, median_coefs_ot_df, perc_25_ot_coefs_df, perc_75_ot_coefs_df)
    
    
    
    # median coefs over time by estimator
    ridge_median_coefs_df1 = ridge_coefs1.apply(lambda x: x.median())
    ridge_median_coefs_df2 = ridge_coefs2.apply(lambda x: x.median())
    ridge_median_coefs_df3 = ridge_coefs3.apply(lambda x: x.median())
    ridge_median_coefs_df4 = ridge_coefs4.apply(lambda x: x.median())
    ridge_median_coefs_df5 = ridge_coefs5.apply(lambda x: x.median())
    ridge_median_coefs_df6 = ridge_coefs6.apply(lambda x: x.median())
    
    lasso_median_coefs_df1 = lasso_coefs1.apply(lambda x: x.median())
    lasso_median_coefs_df2 = lasso_coefs2.apply(lambda x: x.median())
    lasso_median_coefs_df3 = lasso_coefs3.apply(lambda x: x.median())
    lasso_median_coefs_df4 = lasso_coefs4.apply(lambda x: x.median())
    lasso_median_coefs_df5 = lasso_coefs5.apply(lambda x: x.median())
    lasso_median_coefs_df6 = lasso_coefs6.apply(lambda x: x.median())
    
    elnet_median_coefs_df1 = elnet_coefs1.apply(lambda x: x.median())
    elnet_median_coefs_df2 = elnet_coefs2.apply(lambda x: x.median())
    elnet_median_coefs_df3 = elnet_coefs3.apply(lambda x: x.median())
    elnet_median_coefs_df4 = elnet_coefs4.apply(lambda x: x.median())
    elnet_median_coefs_df5 = elnet_coefs5.apply(lambda x: x.median())
    elnet_median_coefs_df6 = elnet_coefs6.apply(lambda x: x.median())
    
    ridge_median_over_time_df = pd.concat([ridge_median_coefs_df1,
                                           ridge_median_coefs_df2,
                                           ridge_median_coefs_df3,
                                           ridge_median_coefs_df4,
                                           ridge_median_coefs_df5,
                                           ridge_median_coefs_df6,
                                          ])[::-1]
    ridge_median_over_time_df['date'] = date
    
    lasso_median_over_time_df = pd.concat([lasso_median_coefs_df1,
                                           lasso_median_coefs_df2,
                                           lasso_median_coefs_df3,
                                           lasso_median_coefs_df4,
                                           lasso_median_coefs_df5,
                                           lasso_median_coefs_df6,
                                          ])[::-1]
    lasso_median_over_time_df['date'] = date
    
    elnet_median_over_time_df = pd.concat([elnet_median_coefs_df1,
                                           elnet_median_coefs_df2,
                                           elnet_median_coefs_df3,
                                           elnet_median_coefs_df4,
                                           elnet_median_coefs_df5,
                                           elnet_median_coefs_df6,
                                          ])[::-1]
    elnet_median_over_time_df['date'] = date
     
    
    # overall coefs
    ridge_overall_coefs_df = pd.concat([ridge_coefs1,
                                        ridge_coefs2,
                                        ridge_coefs3,
                                        ridge_coefs4,
                                        ridge_coefs5,
                                        ridge_coefs6,    
                                       ])
    
    lasso_overall_coefs_df = pd.concat([lasso_coefs1,
                                        lasso_coefs2,
                                        lasso_coefs3,
                                        lasso_coefs4,
                                        lasso_coefs5,
                                        lasso_coefs6,    
                                       ])
    
    elnet_overall_coefs_df = pd.concat([elnet_coefs1,
                                        elnet_coefs2,
                                        elnet_coefs3,
                                        elnet_coefs4,
                                        elnet_coefs5,
                                        elnet_coefs6,    
                                       ])
    
    
    # overall median coefs
    overall_coefs_df = pd.concat([ridge_overall_coefs_df, lasso_overall_coefs_df, elnet_overall_coefs_df])
    
    fig2_overall = plt.figure(figsize=(6,6))
    sns.barplot(data=overall_coefs_df, estimator=np.median, ci=95, orient='h', errwidth=1.5, order=overall_coefs_df.apply(lambda x: x.median()).sort_values(ascending=False).index.tolist(), palette='viridis')
    plt.title('{} overall median coefs with 95% confidence interval'.format(multiple_name))
    plt.tight_layout()
    plt.close()
    
    # overall median coefs by estimator
    ridge_overall_median_coefs_df = pd.DataFrame(ridge_overall_coefs_df.apply(lambda x: x.median()))
    lasso_overall_median_coefs_df = pd.DataFrame(lasso_overall_coefs_df.apply(lambda x: x.median()))
    elnet_overall_median_coefs_df = pd.DataFrame(elnet_overall_coefs_df.apply(lambda x: x.median()))
    
    overall_median_coefs_by_est = pd.concat([ridge_overall_median_coefs_df,
                                             lasso_overall_median_coefs_df,
                                             elnet_overall_median_coefs_df,
                                            ]).reset_index()
    
    ridge_name_series = pd.Series(['ridge']*len(ridge_overall_median_coefs_df))
    lasso_name_series = pd.Series(['lasso']*len(lasso_overall_median_coefs_df)) 
    elnet_name_series = pd.Series(['elnet']*len(elnet_overall_median_coefs_df)) 
    
    estimator_names = pd.concat([ridge_name_series, lasso_name_series, elnet_name_series]).values
    overall_median_coefs_by_est['estimator'] = estimator_names
    overall_median_coefs_by_est.columns = ['residual value driver', 'median coef', 'estimator']
    
    fig3_overall_by_est = plt.figure(figsize=(12,5))
    sns.barplot(x='residual value driver', y='median coef', data=overall_median_coefs_by_est, hue='estimator', palette='viridis', order=overall_coefs_df.apply(lambda x: x.median()).sort_values(ascending=False).index.tolist())
    plt.xticks(rotation=45, ha='right')
    plt.title('{} overall median coefs by estimator'.format(multiple_name))
    plt.tight_layout()
    plt.close()
    
    
    return median_coefs_ot_df, perc_25_ot_coefs_df, perc_75_ot_coefs_df, fig2_overall, fig3_overall_by_est, fig1_list
    

    
def plot_coefs_over_time(multiple_name, df, df25, df75):
    df_t = df.transpose()
    df25_t = df25.transpose()
    df75_t = df75.transpose()
    for col in df_t.columns:
        fig = plt.figure(figsize=(6,2))
        sns.lineplot(data=df_t[col].transpose(), linewidth=1.6, color='g')
        plt.axhline(y=0, color='r', ls='--', linewidth=1.2)
        plt.margins(y=1)
        x = df.columns
        plt.fill_between(x, df25_t[col].transpose().values, df75_t[col].transpose().values, color='g', alpha=.2)
        plt.title('{} median coef over time with 25th and 75th percentile'.format(multiple_name))
        plt.tight_layout()
     
    return fig
























