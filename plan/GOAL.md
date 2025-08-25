# 🎢 Forecasting Goal: Predicting Attraction Return Times
## 1. Goal
I am working on an app to help  plan your day at disneyland. The goal here is to build a model that predicts,
(as accurately as possible) the wait and return times for an attraction.
The model should leverage past patterns (daily/weekly/seasonal) and update predictions as the day progresses.

## 2. Data Setup
* **Target data:**
I have collected per minute data for the past 18 months for all attractions in Disneyland. The data includes:
 1. return time -> For lightning lane attractions, for a given time of the day the return time in minutes is defined as
   a) Before park opening time(6-8am) -> first available return time - park opening time
   b) After park opening time(normally 8am) -> first available return time - current time
 2. wait time -> The wait time in minutes for a given time of the day if you wer to stand in the standby line.

* **Features:**
I have roll ups of this data to the hourly frequency (hourly maxima) plus some engineered features which describe 
  number of minutes attraction was available, return time was available in this hour
  Running totals of these.
  Similar variables for the whole park

* **Features:**
    * I have a list of features that describe the parks state at any given time. See the schema below.
* **Cleaning:** treat nan as 0
---

## 3. Forecasting Task Framing
Initially, I will focus only on the return time prediction.
* **Training window:** last 12 months of data.
Every morning I will train and generate a model on the last 1 year of data and use it to predict the return times.


Throughout the day. At any time that I ask for a prediction, the model should take account of the actuals up to that point
and predict the return times from "now" to park close (typically 12am). ( hourly maximums)

* **Granularity:** hourly predictions.
* **Target:** return_time_mins (minutes). ( Actually max aggregation at hourly level)
* **Input :** Training data at the hourly frequency**.
---

## 6. Experiment and Evaluation Harness
I want to be able to evaluate the efficacy of different modeling strategies and feature sets. I will
build an evaluation harness that allows me to do this.

### Experiment setup
An experiment will be defined by
1. A custom modeling strategy class confirming to the interface 
   DisneyModel(train, test)
    Train and test are pandas dataframes with a datetime index and the features as columns. The model might internally
    split the data into train and validation sets.
   train() -> trains the model
   predict(attraction_name , history) -> prediction for the attraction name given the history dataframe.
   history is to the point in time where the prediction is being made. The output is a pandas series with datetime index
   and target predictions as values. The predictions should be at per minute frequency. Internally the model can interpolate
   from the hourly maxima to per minute frequency. 
   In the future we want to try feeding more granular data and check if the overall accuracy improves.
 
 2. Testing strategy
    Test harness should take a csv file and parse it into a pandas dataframe.
    Define a test period ( last 1 month for example)
        - For each day in the test period:
            - Train the model early morning (say upto 3 am data cutoff point).
            - Predict hourly park open (8 am) to park close (12 am).
    Evaluate using metrics, MAE and MAPE.

    
### Metrics
    Each prediction is done at a infer_hour ( the hour we call the prediction function)
    The prediction is done for the horizons until the end of the day( output is minutely predictions). We can aggregate
     the metrics by the hour of the prediction (pred_hour)

    I need stats per attraction 
        overall MAE, MAE by pred_hour, MAE by infer_hour 

---

## 7. Baseline Model
Just use a very simple naive model to test the harness.

## Features Schema for LL attractions
Exported from bigquery

gw_name STRING, -- attraction name
date DATE, -- date of the sample
pred_hour INT64, -- hour at which the data is rolled up (0-23)
pred_hour_pdt DATETIME, -- pred_hour date time in PDT timezone
mrtd INT64, -- max return time delta
mwt INT64, -- max wait time
mrtd_prev INT64, -- max return time delta previous hour
mrtd_sum_prev INT64, -- sum of max return time delta previous hours
mrtd_park_prev INT64, -- max return time delta previous hour for the park
mrtd_park_sum_prev INT64, -- sum of max return time delta previous hours for the park
mwt_prev INT64, -- max wait time previous hour
mwt_sum_prev INT64, -- sum of max wait time previous hours
mwt_park_prev INT64, -- max wait time previous hour for the park
mwt_park_sum_prev INT64, -- sum of max wait time previous hours for the park
am_prev INT64, -- availability minutes previous hour
am_sum_prev INT64, -- sum of availability minutes previous hours
am_park_prev INT64, -- availability minutes previous hour for the park
am_park_sum_prev INT64, -- sum of availability minutes previous hours for the park
om_prev INT64, -- operating minutes previous hour
om_sum_prev INT64, -- sum of operating minutes previous hours
om_park_prev INT64, -- operating minutes previous hour for the park
om_park_sum_prev INT64 -- sum of operating minutes previous hours for the park


