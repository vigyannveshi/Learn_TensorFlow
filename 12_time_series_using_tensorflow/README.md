### ↩️ [**Learn Tensorflow**](../README.md)

### Content
| <u>**12. Time Series forecasting using Tensorflow**</u> ||
|---------|----------|
| **Concept** | **Notebook** |
|01. Data loading and pre-processing|[01](01_time_series_using_tensorflow.ipynb)|
|02. Alternative to import data using python-csv module |[01](01_time_series_using_tensorflow.ipynb)|
|03. Creating train-test splits for time-series data (wrong-way)|[01](01_time_series_using_tensorflow.ipynb)|
|04. Creating train-test splits for time-series data (right-way)|[01](01_time_series_using_tensorflow.ipynb)|
|05. Creating plotting function to visualize time-series data |[01](01_time_series_using_tensorflow.ipynb)|
|06. Modelling Experiments |[02](02_time_series_using_tensorflow.ipynb)|
|07. Building Naive model (baseline) |[02](02_time_series_using_tensorflow.ipynb)|
|08. Most common time series evaluation metrics |[02](02_time_series_using_tensorflow.ipynb)|
|09. Implementing MASE in code |[02](02_time_series_using_tensorflow.ipynb)|
|10. Creating function to evaluate model's forcast |[02](02_time_series_using_tensorflow.ipynb)|
|11. Other models for time series forcasting|[02](02_time_series_using_tensorflow.ipynb)|
|12. Windowing our dataset|[03](03_time_series_using_tensorflow.ipynb)|
|13. Preprocessing function to window our dataset|[03](03_time_series_using_tensorflow.ipynb)|
|14. Turning windows into training and test sets|[03](03_time_series_using_tensorflow.ipynb)|
|15. Creating a modelling checkpoint callback to save our model|[04](04_time_series_using_tensorflow.ipynb)|
|16. Building Dense model (model 1) (horizon = 1, window = 7)|[04](04_time_series_using_tensorflow.ipynb)|
|17. Building Model 2 (same as model 1) (horizon = 1, window = 30)|[05](05_time_series_using_tensorflow.ipynb)|
|18. Building Model 3 (same as model 1) (horizon = 7, window = 30)|[06](06_time_series_using_tensorflow.ipynb)|
|19. Comparing modelling experiments (baseline,1,2,3)|[07](07_time_series_using_tensorflow.ipynb)|
|20. Preparing data for input to sequence models |[08](08_time_series_using_tensorflow.ipynb)|
|21. Building Model 4 (conv1D) (horizon = 1, window = 7)|[08](08_time_series_using_tensorflow.ipynb)|
|22. Building Model 5 (LSTM) (horizon = 1, window = 7)|[09](09_time_series_using_tensorflow.ipynb)|
|23. Multivariate time series (Model 6)|[10](10_time_series_using_tensorflow.ipynb)|
|24. Making a windowed dataset using Pandas|[10](10_time_series_using_tensorflow.ipynb)|
|25. Replicating N-BEATS algorithm (Model 7)|[11](11_time_series_using_tensorflow.ipynb)|
|25.1. Building and testing the N-BEATS block layer|[11](11_time_series_using_tensorflow.ipynb)|
|25.2. Creating a performant data pipeline using `tf.data` |[11](11_time_series_using_tensorflow.ipynb)|
|25.3. Setting up hyperparameters for N-BEATS (Generic version) with Daily data |[11](11_time_series_using_tensorflow.ipynb)|
|25.4. Setting up residual connections in N-BEATS |[11](11_time_series_using_tensorflow.ipynb)|
|25.5. Building, Compiling and fitting the N-Beats algorithm |[11](11_time_series_using_tensorflow.ipynb)|
|25.6. Saving and loading N-BEATS model |[11](11_time_series_using_tensorflow.ipynb)|
|25.7. Plotting the N-BEATS architecture  |[11](11_time_series_using_tensorflow.ipynb)|
|26. Creating an ensemble (Model 8) |[12](12_time_series_using_tensorflow.ipynb)|
|27. Saving and loading trained ensemble model |[12](12_time_series_using_tensorflow.ipynb)|
|28. Importance of prediction intervals (uncertainity estimates) in forcasting |[12](12_time_series_using_tensorflow.ipynb)|
|29. Types of uncertainity in machine learning |[13](13_time_series_using_tensorflow.ipynb)|
|30. Future Prediction Model (Model 9)|[13](13_time_series_using_tensorflow.ipynb)|
|31. Black Swan Theory - The turkey problem (Model 10) (same as model 1) (horizon = 1, window = 7) |[14](14_time_series_using_tensorflow.ipynb)|
|32. Comparing the models trained so far |[15](15_time_series_using_tensorflow.ipynb)|
 

### Notes (Introduction)
* data has time component (temporal)
* Example time series problems:
  * classification (given a time varying signal) [output: discrete]
    * eg: heart beat monitoring (electro-cardiogram), to check for anamolies?
    * eg: what electronic device is it?
  * Forcasting [output: continuous]
    * The longer you forcast in the future, the higher the chance of an error
    * how much?
    * how many?
    * market demand?
    * compute demands
    * market campaigns
    * other forcasting problems:
      * weather
      * food quantity we need
      * estimating how long something's gonna take
* What can we forcast?
  * Things which have the most impact in our lives are usually unplanned and un-predictable
  * Although we can build a model to make predictions, but it is not necessary for every prediction one makes to be true
  * https://otexts.com/fpp3/
  * https://arxiv.org/pdf/1905.10437 | N-BEATS: NEURAL BASIS EXPANSION ANALYSIS FOR
INTERPRETABLE TIME SERIES FORECASTING
* We can have black swan events in time-series prediction/forcasting
* Inputs and Outputs of our forcasting problem:
  * output: continuous
  * input: time-series data is already numerical
* for bit-predict problem:
  * input shape: [batch_size,window_size]
  * output shape: [horizon]
  * window is the amount of data we put into a model for a time series
  * windows are changeable
  * horizon is the time for which the output is predicted for
  * It is a sequence problem (seq2seq)
    * many to one (predicting next day based on previous week's data)
    * many to many (predicting next week based on previous month's data)
* Different kinds of time-series patterns
  
  <img src ='time_series_patterns.png'></img>
  * `Trend`: A trend exists when there is a long-term increase or decrease in the data. It does not have to be linear. Sometimes we will refer to a trend as “changing direction”, when it might go from an increasing trend to a decreasing trend.There is a trend in the antidiabetic drug sales data shown in Figure 2.2.
  * `Seasonal`: A seasonal pattern occurs when a time series is affected by seasonal factors such as the time of the year or the day of the week. Seasonality is always of a fixed and known period. The monthly sales of antidiabetic drugs (Figure 2.2) shows seasonality which is induced partly by the change in the cost of the drugs at the end of the calendar year. (Note that one series can have more than one seasonal pattern.)
  * `Cyclic`: A cycle occurs when the data exhibit rises and falls that are not of a fixed frequency. These fluctuations are usually due to economic conditions, and are often related to the “business cycle”. The duration of these fluctuations is usually at least 2 years.
  
* Univariate and Multi-variate time-series data
  * Univariate: one-variable (eg: using price of bitcoin predict price of bitcoin)
  * Multi-variate: more than one variable (eg: using price of bitcoin + block reward size to predict price of bitcoin)

    
