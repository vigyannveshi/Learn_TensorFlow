
# TIME SERIES FUNCTIONALITIES

### ------- FUNCTIONS -------###
# 1. get_labelled_window() 
# 2. make_windows()
# 3. make_train_test_splits()
# 4. mean_absolute_scaled_error()
# 5. evaluate_time_series()
# 6. plot_time_series()
# 7. make_preds()
# 8. make_ensemble_preds()
# 9. get_upper_lower_bounds()
# 10. make_future_forecasts()
# 11. 
### ------- FUNCTIONS -------###


### ------- CONSTANTS -------###
HORIZON = 1
WINDOW_SIZE = 7
### ------- CONSTANTS -------###


### IMPORTANT IMPORTS ###
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
### IMPORTANT IMPORTS ###

# 1. to label windowed data
def get_labelled_windows(x,horizon = HORIZON):
    '''
    Creates labels for windowed dataset

    E.g. if horizon = 1
    Input: [[0,1,2,3,4,5,6,7] 
            [1,2,3,4,5,6,7,8]
            [2,3,4,5,6,7,8,9],...]
    
    Output: [([0,1,2,3,4,5,6],[7])
    '''

    # the first colon is for all rows
    return x[:,:-horizon],x[:,-horizon:]



# 2. creating windows and labels from time-series array
def make_windows(x,window_size = WINDOW_SIZE, horizon = HORIZON ):
    '''
    Turns a 1D array into a 2D array of sequential labelled windows of window_size with horizon size labels.  
    '''
    # 1. create window step of particular window_size (add the horizon on the end for labelling later)
    window_step = np.expand_dims(np.arange(window_size + horizon),axis=0)

    # 2. use numpy indexing to create a 2D array of multiple windows (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)),axis = 0).T

    # 3. index on target array (a time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. get the labelled windows
    windows,labels = get_labelled_windows(windowed_array,horizon=horizon)
    return windows,labels



# 3. make the train-test splits
def make_train_test_splits(windows,labels,test_split=0.2):
    '''
    Splits matching pair of windows and labels into train and test splits.
    '''

    split_size = int((1-test_split)*len(windows))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]

    return train_windows,test_windows,train_labels,test_labels

# 4. Mean Absolute Scaled Error
def mean_absolute_scaled_error(y_true,y_pred,seasonality = 1):
    '''
    Implement MASE (assuming non-seasonality of the data)

    Parameters
    ----------
    y_true: true values
    y_pred: predicted values
    seasonality: min-time in time sequence, default: 1
    '''
    mae = tf.reduce_mean(tf.abs(y_true-y_pred))

    # find MAE of naive forcast (no-seasonality)
    mae_naive = tf.reduce_mean(tf.abs(y_true[seasonality:] - y_true[:-seasonality]))

    return mae/mae_naive

# 5. evaluate time-series results
def evaluate_time_series(y_true, y_pred):
    # make sure float32 datatype for metric calculation
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # calculate various evaluation metrics
    mae = tf.keras.metrics.mae(y_true, y_pred)
    mse = tf.keras.metrics.mse(y_true,y_pred)

    # for horizon > 1, we will get vector outputs, need to further mean_reduce them
    if mae.ndim > 0:
        mae = tf.reduce_mean(mae)
    if mse.ndim > 0:
        mse = tf.reduce_mean(mse)

    rmse = tf.sqrt(mse)

    mape = tf.keras.metrics.mape(y_true,y_pred)
    if mape.ndim>0:
        mape = tf.reduce_mean(mape)

    mase = mean_absolute_scaled_error(y_true,y_pred)


    return {
        "mae": mae.numpy(),
        "mse": mse.numpy(),
        "rmse": rmse.numpy(),
        "mape": mape.numpy(),
        "mase": mase.numpy(), 
    }


# 6. plot_time_series
def plot_time_series(timesteps,values,format='.',start=0,end=None,label=None,xlabel='Time',ylabel='Value',grid=True,legend_text_size=10,label_text_size=10,xtick_rotation=0,prediction_interval = False,**kwargs):
    '''
    Plots timesteps (a series of points in time) against values (a series of values across timesteps).
    
    Parameters 
    ----------  
    timesteps : array of timestep values 
    values: array of values across time
    format: style of plot, default "."
    start: where to start the plot (setting a value will index from start of timesteps & values), default: 0
    end: where to end the plot (setting a value will end at particular index of timesteps & values), default: None
    label: label to show on plot about values, default: None
    xlabel: sets the x-label, default "Time"
    ylabel: sets the y-label, default "Values" 
    '''

    # plot the series:
    if not prediction_interval:
        plt.plot(timesteps[start:end],values[start:end],format,label=label,**kwargs)
    else:
        plt.fill_between(timesteps[start:end],values[0][start:end],values[1][start:end],label = label,**kwargs)
    plt.xlabel(xlabel,fontsize=label_text_size)
    plt.ylabel(ylabel,fontsize=label_text_size)
    plt.xticks(rotation=xtick_rotation)
    if label:
        plt.legend(fontsize=legend_text_size)
    plt.grid(grid)



# 7. make prediction using a model
def make_preds(model,input_data):
    '''
    Uses model to make predictions on input data
    '''

    forcast = model.predict(input_data)
    return tf.squeeze(forcast) # return 1D array of predictions



# 8. make prediction using an ensemble of models
def make_ensemble_preds(ensemble_models,input_data):
    '''
    Makes prediction using an ensemble of models
    '''
    ensemble_preds = []

    for model in ensemble_models:
        preds = model.predict(input_data)
        ensemble_preds.append(preds)
        
    return tf.constant(tf.squeeze(ensemble_preds))


# 9. finding upper and lower bounds of ensemble predictions
def get_upper_lower_bounds(preds,z_value = 1.96): # 1. Take predictions from a number of randomly initialized models
    # 2. Measure the standard deviation of predictions
    std = tf.math.reduce_std(preds,axis = 0) 
    
    # 3. Multiply the standard deviation by 1.96
    standard_error = z_value * std

    # 4. Get the prediction interval 
    preds_mean = tf.reduce_mean(preds, axis = 0)
    lower,upper = preds_mean-standard_error, preds_mean + standard_error

    return lower,upper


# 10. making future predictions using model and sequence of values
def make_future_forecasts(values,model,into_future, window_size =WINDOW_SIZE, train_dataset = None,retrain = False, epochs = 10) -> list:
    '''
    Makes future forecasts into_future steps after values ends.
    
    Returns future forecasts as a list of floats.
    '''

    # create an empty list
    future_forecasts = []
    last_window = values[-window_size:]

    if retrain:
        train_data = copy.deepcopy(train_dataset[0])
        train_labels = copy.deepcopy(train_dataset[1])
        model = copy.deepcopy(model)

    # make into_future number of predictions, altering the data which gets predicted on each turn
    for _ in range(into_future):
        # predict on last window, then append it again, again, again...
        # model will eventually start to forecast using its own forecasts
        future_pred = model.predict(tf.expand_dims(last_window,axis = 0),verbose=0)
        print(f'Predicting on:\n{last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n')

        # append the predictions to future_forcast
        future_forecasts.append(tf.squeeze(future_pred).numpy())

        # update the last window with new pred, and get WINDOW_SIZE most recent preds
        last_window = np.append(last_window,future_pred)[-window_size:]

        # retrain model everytime after prediction
        if retrain == True:
            print(f"Model retraining for {epochs} epochs before next prediction")
            window2append = np.append(train_data[-1],train_labels[-1])[-window_size:]
            
            train_data = np.append(train_data,np.expand_dims(window2append,axis = 0),axis = 0)
            train_labels =np.append(train_labels,tf.squeeze(future_pred).numpy())
        
            model.fit(train_data,train_labels,
            epochs = epochs,
            verbose = 0
            ) 
    return future_forecasts


# 11. get a sequence of dates from starting dates upto dates into_future
def get_future_dates(start_date,into_future,offset=1):
    '''
    Returns array of date-time values ranging from start_date to start_date + into_future
    '''
    return np.array([start_date + np.timedelta64(i,'D') for i in range(1,into_future+1,offset)],dtype='datetime64[D]') # returns a date-range between start and end date