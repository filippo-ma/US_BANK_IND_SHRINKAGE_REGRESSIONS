# Valuation dates

# US companies have 45 calendar days (or 40 days for large accelerated filers and accelerated filers) to report 
# previous quarter data (SEC Form 10-Q) (at most 90 days for 10-K reports). Companies can, however, get an extension (5d) 
# and file after 45 days but probably not miss a whole quarter. That being so, the following dates:

from datetime import datetime, timedelta

valuation_datetime1 = datetime(2020,10,1)
valuation_datetime2 = datetime(2020,4,1)
valuation_datetime3 = datetime(2019,10,1)
valuation_datetime4 = datetime(2019,4,1)
valuation_datetime5 = datetime(2018,10,1)
valuation_datetime6 = datetime(2018,4,2)
valuation_date1 = valuation_datetime1.strftime('%Y-%m-%d')
valuation_date2 = valuation_datetime2.strftime('%Y-%m-%d')
valuation_date3 = valuation_datetime3.strftime('%Y-%m-%d')
valuation_date4 = valuation_datetime4.strftime('%Y-%m-%d')
valuation_date5 = valuation_datetime5.strftime('%Y-%m-%d')
valuation_date6 = valuation_datetime6.strftime('%Y-%m-%d')

# last qtr end date at valuation date
val_date1_last_qtr_date = '2020-06-30'
val_date2_last_qtr_date = '2019-12-31'
val_date3_last_qtr_date = '2019-06-30'
val_date4_last_qtr_date = '2018-12-31'
val_date5_last_qtr_date = '2018-06-30'
val_date6_last_qtr_date = '2017-12-31'