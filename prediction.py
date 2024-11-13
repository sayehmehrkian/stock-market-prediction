
import os
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

# Set TensorFlow log level to only errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Suppress other warnings
warnings.filterwarnings("ignore")


# Suppress warnings and future deprecations from pandas
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)



from datetime import datetime, timedelta
import matplotlib.pyplot as plt


from persiantools.jdatetime import JalaliDate
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from models import *

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

import random

# Set a global seed value
seed = 42

# Set seeds for reproducibility
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


parser = argparse.ArgumentParser()

parser.add_argument('-t', '--typ', type=str, nargs='+')
parser.add_argument('-m', '--model', type=str, help='LSTM or Transformers')
parser.add_argument('-e', '--epochs', default=4, type=int)
parser.add_argument('-tr', '--threshold', type=float, default=0.95)
parser.add_argument('-ttr', '--train_test_ratio', type=float, default=0.85, help="the split ratio of the ")
parser.add_argument('-p', '--passwd', type=str, default="1q9ze6QA8", help="password is needed to connect")

args = parser.parse_args()

train_test_ratio = args.train_test_ratio


def filter_threshold(val):
  # print(val, type(val))
  if max(val) < args.threshold:
    return np.nan
    # return np.nan
  else:
    return val


def convert_str2float(val):
  val = val.replace('[', '')
  val = val.replace(']', '')
  val = [round(float(i), 4) for i in val.split(',')]
  return val


def jalali_2_greg(jdate):
  jdate_list = list(str(jdate))
  jyear = ''.join(jdate_list[0: 4])
  jmon = ''.join(jdate_list[4: 6])
  jday = ''.join(jdate_list[6:])
  return JalaliDate(int(jyear), int(jmon), int(jday)).to_gregorian()


def concat_dfs(dirc, no=30):
  df_list = [pd.read_excel(f"{dirc}/out_{i}.xlsx") for i in range(no)]
  # pd.concat(df_list).to_excel(f'{address}/commet_labeled.xlsx')
  return pd.concat(df_list)


# read shakhes
indx_val_rev = pd.read_excel('./IndexData1.xlsx')

# Make shakhes incremental, Old to New dates.
indx_val = indx_val_rev.loc[::-1].reset_index(drop=True)

print('\n Trading Days based on IndexData.xlsx  :', indx_val.shape[0])
# turn jalali to gregorian calendar
indx_val['datetime'] = indx_val['dateissue'].apply(jalali_2_greg)

# aggregated dataframe of sentiments
df_sent = concat_dfs(dirc='./output')
# print('Reading all the dfs DONE! df shape', df_sent.shape)
tpeix = pd.read_excel('./TEPIX.xlsx')
print("TEPIX.xlsx , the whole number of comments about index (shapes)", tpeix.shape)
print('\n\nThe number of comments entries (>7 & <50), \nremoved the short length and keeped the right sized : ',
      df_sent.shape[0])
print(f"It starts from {indx_val['datetime'].iloc[0]} to {indx_val['datetime'].iloc[-1]}")
# print(f'It starts from {df[']}')
df_sent = df_sent.sort_values('datetime', ascending=False).reset_index(drop=True)
df_sent['pred_scores'] = df_sent['pred_scores'].apply(convert_str2float)
df_sent['pred_scores'] = df_sent['pred_scores'].apply(filter_threshold)
df = df_sent.dropna(axis=0, subset=['pred_scores'])

print('\nAfter applying threshold, The number will be ', df.shape[0])
print('We use this dataframe and chunked each day with the corresponding index price')

noon = datetime(2021, 8, 22, hour=13).time()
# print('noon is ', noon)

datalist = []
commentlist = []
shape_list = []

print(indx_val.shape)

for i, row in indx_val.iterrows():
  if i < 297:
    #     print(i, '   ', row['datetime'])
    # four lines below will be deleted
    # row['datetime'] = row['datetime'].replace(month=1, day=23)
    # indx_val['datetime'].iloc[i + 1] = indx_val['datetime'].iloc[i + 1].replace(month=1, day=22)
    # print('in the loop ', indx_val['datetime'].iloc[i + 1])
    start_dt = datetime.combine(row['datetime'], noon)
    end_dt = datetime.combine(indx_val['datetime'].iloc[i + 1], noon)

    # print('end and start time ', end_dt, start_dt)
    sent_spec_df = df[df['datetime'].between(start_dt, end_dt)]
    datalist.append(row['Value'])
    commentlist.append(sent_spec_df['pred_scores'].values.tolist())
    shape_list.append(sent_spec_df['pred_scores'].shape[0])

max_shape = np.max(shape_list)
min_shape = np.min(shape_list)
mean_shape = np.mean(shape_list)
print('\n\nBecause the sentiment array are not same, due to different number of')
print('commnets in each day, we padded all the array by zero to maximum number')
print(f'of comments\n')
print(f'The max  number of comments in each day is {max_shape}.')
print(f'The min  number of comments in each day is {min_shape}.')
print(f'The mean number of comments in each day is {mean_shape}.')

# Convert the dataframe to a numpy array
datalist = np.array(datalist)
commentlist = np.asarray([np.array(x) for x in commentlist], dtype='object')
lenght_ = datalist.shape[0]


# print('comment list sahpe ', commentlist.shape)

def to_shape(x, shape):
  if x.shape[0] != shape:
    # print('padding zero to the start of the tensor')
    z = np.zeros((shape, 3))
    z[(shape - x.shape[0]): shape, :] = x
    return z
  else:
    return x


commentlist = np.array([to_shape(x, max_shape) for x in commentlist])

# # dataset = data.values
# # Get the number of rows to train the model on

training_data_len = int(np.ceil(lenght_) * train_test_ratio)
# print('train data length ', training_data_len)
# # # Scale the data

datalist = datalist.reshape((-1, 1))
scaler = MinMaxScaler(feature_range=(0.01, 0.99))
scaled_data = scaler.fit_transform(datalist)


train_data = scaled_data[0:int(training_data_len)]
train_comment = commentlist[0:int(training_data_len)]
# print('train_data shape ', train_data.shape, train_comment.shape)

# Split the data into x_train and y_train data sets
x1_train = []
x2_train = []
y_train = []

if 'comment' in args.typ:
  print('\n\nTrain shapes with comments ---- data, comments ', train_data.shape, train_comment.shape)
if 'nocomment' in args.typ:
  print('\n\nTrain shapes without comments', train_data.shape)

# Batching
for i in range(7, len(train_data) - 7):
  x1_train.append(train_data[i - 7:i, :])
  x2_train.append(train_comment[i - 7:i, :, :])
  y_train.append(train_data[i + 6, 0])

# print(len(x1_train), len(x2_train), len(y_train))


# Convert the x_train and y_train to numpy arrays
x1_train, x2_train, y_train = np.array(x1_train), np.array(x2_train), np.array(y_train)
# print('x2 train shape ', x2_train.shape)
x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1], 1))
x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1], x2_train.shape[2], 3))


# *****************************************************************************
# *****************************************************************************

inpt1 = Input(shape=(x1_train.shape[1:]), name="price")
inpt2 = Input(shape=(x2_train.shape[1:]), name="sentiments")
print('inpt1 & inpt2 shape ', inpt1.shape, inpt2.shape)


# *****************************************************************************

if args.model == 'LSTM':

  if 'nocomment' in args.typ:
    pred = lstm_nocomment(inpt1)
    mymodel = Model(inputs=inpt1, outputs=pred)
    print(f'My model is {args.model} without Comments\n', mymodel.summary())

  elif 'comment' in args.typ:
    pred = lstm_comment(inpt1, inpt2)
    mymodel = Model(inputs=[inpt1, inpt2], outputs=pred)
    print(f'My model is {args.model} with adding Comments\n', mymodel.summary())


# *****************************************************************************

elif args.model == 'Transformers':

  if 'nocomment' in args.typ:
    print('Transformers nocomment  ', inpt1.shape)
    pred = build_transformer(inpt1, inpt2=None, head_size=64,
                                    num_heads=4, ff_dim=2, num_trans_blocks=2,
                                    mlp_units=[256], mlp_dropout=0.10,
                                    dropout=0.10, attention_axes=1)

    mymodel = Model(inputs=inpt1, outputs=pred)
    print(f'My model is {args.model} without Comments\n', mymodel.summary())
  
  elif 'comment' in args.typ:
    pred = build_transformer(inpt1, inpt2, head_size=128,
                                    num_heads=4, ff_dim=2, num_trans_blocks=4,
                                    mlp_units=[256], mlp_dropout=0.10,
                                    dropout=0.10, attention_axes=1)

    mymodel = Model(inputs=[inpt1, inpt2], outputs=pred)
    print(f'My model is {args.model} with Comments\n', mymodel.summary())

# *****************************************************************************

elif args.model == "GRU":
  if 'nocomment' in args.typ:
    pred = gru_nocomment(inpt1)
    mymodel = Model(inputs=inpt1, outputs=pred)
    print(f'My model is {args.model} without Comments\n', mymodel.summary())

  elif 'comment' in args.typ:
    pred = gru_comment(inpt1, inpt2)
    mymodel = Model(inputs=[inpt1, inpt2], outputs=pred)
    print(f'My model is {args.model} with adding Comments\n', mymodel.summary())

# *****************************************************************************

elif args.model == "FFNN":
  if 'nocomment' in args.typ:
    pred = ffnn_nocomment(inpt1)
    mymodel = Model(inputs=inpt1, outputs=pred)
    print(f'My model is {args.model} without Comments\n', mymodel.summary())

  elif 'comment' in args.typ:
    pred = ffnn_commnet(inpt1, inpt2)
    mymodel = Model(inputs=[inpt1, inpt2], outputs=pred)
    print(f'My model is {args.model} with adding Comments\n', mymodel.summary())

# *****************************************************************************

elif args.model == "CNN_LSTM":
  if 'nocomment' in args.typ:
    pred = cnn_lstm_nocomment(inpt1)
    mymodel = Model(inputs=inpt1, outputs=pred)
    print(f'My model is {args.model} without Comments\n', mymodel.summary())

  elif 'comment' in args.typ:
    pred = cnn_lstm_comment(inpt1, inpt2)
    mymodel = Model(inputs=[inpt1, inpt2], outputs=pred)
    print(f'My model is {args.model} with adding Comments\n', mymodel.summary())

# *****************************************************************************

elif args.model == "Bi_LSTM":
  if 'nocomment' in args.typ:
    pred = bi_lstm_nocomment(inpt1)
    mymodel = Model(inputs=inpt1, outputs=pred)
    print(f'My model is {args.model} without Comments\n', mymodel.summary())

  elif 'comment' in args.typ:
    pred = bi_lstm_comment(inpt1, inpt2)
    mymodel = Model(inputs=[inpt1, inpt2], outputs=pred)
    print(f'My model is {args.model} with adding Comments\n', mymodel.summary())

# *****************************************************************************

# *****************************************************************************

# Compile the model
mymodel.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
if 'nocomment' in args.typ:
  # without comments
  mymodel.fit(x1_train, y_train, batch_size=1, epochs=args.epochs)
elif 'comment' in args.typ:
  # with comments
  mymodel.fit([x1_train, x2_train], y_train, batch_size=1, epochs=args.epochs)

# Create the testing data set
test_data = scaled_data[training_data_len - 7:]
train_comment = commentlist[training_data_len - 7:]

# Create the data sets x_test and y_test
x1_test = []
x2_test = []
y_test = datalist[training_data_len:, :]

for i in range(7, len(test_data)):
  x1_test.append(test_data[i - 7:i, 0])
  x2_test.append(commentlist[i - 7:i, :, :])

# Convert the data to a numpy array
x1_test = np.array(x1_test)
x2_test = np.array(x2_test)

# Reshape the data
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1], 1))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1], x2_test.shape[2], 3))

# Get the models predicted price values
print('\n\n prediction step ')
if 'nocomment' in args.typ:
  predictions = mymodel.predict(x1_test)
elif 'comment' in args.typ:
  predictions = mymodel.predict([x1_test, x2_test])

# print('\n error  ', predictions.shape)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print('\n\n   RMSE :    ', rmse)
# print('   R2_Score :  ', r2_score(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
print('   MAE = ', mae)


# MAPE
mape = mean_absolute_percentage_error(y_test, predictions)
print("   MAPE : ", mape)

# Plot the data
train = datalist[:training_data_len]
valid = pd.DataFrame(np.zeros((datalist.shape[0], 2)), columns=['Value', 'Predictions'])

valid['Value'].loc[training_data_len:] = datalist[training_data_len:, 0]
valid['Predictions'].loc[training_data_len:] = predictions.squeeze().tolist()
valid['datetime'] = indx_val['datetime']
valid[valid.eq(0)] = np.nan

# Visualize the data
valid = valid.set_index('datetime')
indx_val = indx_val.set_index('datetime')

plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.xlim(indx_val.index[0], indx_val.index[-5])
plt.ylabel('Index Value ', fontsize=18)
plt.plot(indx_val['Value'])
plt.plot(valid[['Value', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')


if not os.path.exists('./output_img'):
      os.mkdir('./output_img')

if 'nocomment' in args.typ:
  plt.savefig(f'./output_img/{args.model}_noComment.png')
elif 'comment' in args.typ:
  plt.savefig(f'./output_img/{args.model}_Comments.png')

# Show the valid and predicted prices
# print(valid)
print('Done!')


