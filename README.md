# major_project2_nifty


## python code:

```py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import math
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
dataset = pd.read_csv('NIFTY 50_with_indicators_.csv', low_memory=False)
dataset.head()

import pandas as pd
import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(x=dataset['date'][:10],
                open=dataset['open'][:100],
                high=dataset['high'][:100],
                low=dataset['low'][:100],
                close=dataset['close'][:100])])

fig.show()

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2, shuffle = False)

def standardize(data, mean=1, std=1):
    standardized_data = list()
    data = list(data)
    for x in data:
        standardized_data.append((x-mean)/std)
    return standardized_data

def standardize_inverse(data, mean=1, std=1):
    reverse_data = list()
    data = list(data)
    for y in data:
        reverse_data.append((y*std)+mean)
    return reverse_data

all_columns = list(dataset.columns[1:])
means, standard_devs = [], []

for i in range(len(all_columns)):
    column_name = all_columns[i]
    means.append(np.mean(train[column_name].values))
    standard_devs.append(np.std(train[column_name].values))

df_train2 = pd.DataFrame(columns = all_columns)
df_test2 = pd.DataFrame(columns= all_columns)

for i in range(len(all_columns)):
    column_name = all_columns[i]
    df_train2[column_name] = standardize(train[column_name].values, means[i], standard_devs[i])

for i in range(len(all_columns)):
    column_name = all_columns[i]
    df_test2[column_name] = standardize(test[column_name].values, means[i], standard_devs[i])

from pandas import concat
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

data_train = train_reframed.values
data_test = test_reframed.values
trainX, trainy = data_train[:, :n_features*look_back], data_train[:, -1]
testX, testy = data_test[:, :n_features*look_back], data_test[:, -1]
trainX = trainX.reshape(trainX.shape[0], look_back, n_features)
testX = testX.reshape(testX.shape[0], look_back, n_features)
print('Shape of Training Data: ', trainX.shape, trainy.shape)
print('Shape of Test Data: ', testX.shape, testy.shape)

trainX = trainX.reshape(trainX.shape[0], look_back, n_features)
testX = testX.reshape(testX.shape[0], look_back, n_features)

print('Shape of Training Data: ', trainX.shape, trainy.shape)
print('Shape of Test Data: ', testX.shape, testy.shape)

np.random.seed(7)
model = Sequential()
model.add(LSTM(512, input_shape = (trainX.shape[1], trainX.shape[2]), return_sequences=True, activation='relu'))
model.add(LSTM(512, return_sequences=True, activation='relu'))
model.add(LSTM(512, return_sequences=False, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()

history = model.fit(trainX, trainy, epochs = 50, batch_size = 256, verbose = 1, shuffle=False,
                     validation_split=0.1)

model.save("/content/drive/MyDrive/majorproject2model.h5")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
testPredict = model.predict(testX)

testPredict2 = standardize_inverse(testPredict, means[-1], standard_devs[-1])
testy2 = standardize_inverse(testy, means[-1], standard_devs[-1])

testScore = math.sqrt(mean_squared_error(testy2, testPredict2))
print('Test Score: %.2f RMSE' % (testScore))

plt.clf()
plt.figure(figsize=(16, 8))
plt.rcParams.update({'font.size': 16})
plt.plot(actual[:], label = 'Actual')
plt.plot(predicted[:], '--',label = 'Predicted')
plt.title('LSTM model Nifty Price Forecast in 5mins')
plt.ylabel('Close Price [in Rupees]')
plt.xlabel('Time Steps [5 - minutes]')
plt.legend()
plt.show()

```


## OUTPUT:

![image](https://github.com/curiouzs/major_project2_nifty/assets/75234646/93fd85c3-0dfa-4da8-b127-2e2ccc67f1f7)

![image](https://github.com/curiouzs/major_project2_nifty/assets/75234646/08cd4dfe-7dc6-45ba-940b-a74b58865a7d)

![image](https://github.com/curiouzs/major_project2_nifty/assets/75234646/38758da6-f085-4b0c-97d8-9f7a5bcfac8a)
