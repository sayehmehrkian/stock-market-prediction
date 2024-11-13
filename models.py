import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense, LSTM, Conv1D, GRU
from tensorflow.keras.layers import Dropout, MaxPooling1D, Bidirectional
from tensorflow.keras.layers import Concatenate, Flatten
# from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention


# **************************** Transformer *************************
def transformer_encoder(inputs, head_size, num_heads,
                        ff_dim, dropout=0,
                        epsilon=1e-6, attention_axes=None,
                        kernel_size=1):
  """
  Transformer block.
  """
  x = layers.LayerNormalization(epsilon=epsilon)(inputs)
  x = layers.MultiHeadAttention(
    key_dim=head_size, num_heads=num_heads, dropout=dropout,
    attention_axes=attention_axes)(x, x)
  x = Dropout(dropout)(x)

  res = x + inputs

  # Feed Forward Part
  x = layers.LayerNormalization(epsilon=epsilon)(res)
  x = Conv1D(filters=ff_dim, kernel_size=kernel_size, activation="relu")(x)
  x = Dropout(dropout)(x)
  x = Conv1D(filters=inputs.shape[-1], kernel_size=kernel_size)(x)

  return x + res


def build_transformer(inpt1, inpt2, head_size, num_heads, ff_dim,
                      num_trans_blocks, mlp_units, dropout=0,
                      mlp_dropout=0, attention_axes=None,
                      epsilon=1e-6, kernel_size=1):
  """
  Whole model
  """
  n_timesteps, n_features, n_outputs = inpt1.shape[1], inpt1.shape[2], 1

  x1, x2 = inpt1, inpt2
  # print('x1 , x2 shapes at the frist in transormers ', x1.shape, x2.shape)

  for _ in range(num_trans_blocks):
    x1 = transformer_encoder(x1, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
                             attention_axes=attention_axes, kernel_size=kernel_size, epsilon=epsilon)
  if inpt2 is not None:

    x2 = TimeDistributed(LSTM(7, return_sequences=False))(inpt2)
    x = Concatenate()([x1, x2])
    x = Flatten()(x)
  else:
    x = Flatten()(x1)

  for dim in mlp_units:
    x = Dense(dim, activation="relu")(x)
    x = Dropout(mlp_dropout)(x)

  outputs = Dense(n_outputs, activation='sigmoid')(x)
  return outputs

# **************************** LST *************************

def lstm_nocomment(inpt):
  print('\n\ntwo-layer lstm without comment \n\n')
  x1 = LSTM(32, return_sequences=True)(inpt)
  x1 = LSTM(16, return_sequences=False)(x1)
  x1 = Flatten()(x1)
  return Dense(1, activation='sigmoid')(x1)


def lstm_comment(inpt1, inpt2):
  print('\n\nlstm with comment \n\n')
  x1 = LSTM(1, return_sequences=True)(inpt1)
  x1 = Dropout(0.2)(x1)
  # x1 = LSTM(50, return_sequences=False)(x1)
  # x1 = Dropout(0.2)(x1)
  x1 = Flatten()(x1)
  x1 = Dense(10)(x1)
  print(' ejra   61 ')
  x2 = TimeDistributed(LSTM(1, return_sequences=False))(inpt2)
  # x2 = LSTM(1, return_sequences=True)(x2)

  x2 = Flatten()(x2)
  # x2 = Dense(1)(x2)

  conc = Concatenate()([x1, x2])
  return Dense(1, activation='sigmoid')(conc)


# **************************** GRU *************************

def gru_nocomment(inpt):
  print('  exp    17 ')
  x = GRU(50, return_sequences=True, input_shape=inpt.shape)(inpt)
  x = Dropout(0.2)(x)
  x = GRU(10, return_sequences=True)(x)
  x = Dropout(0.2)(x)
  x = Flatten()(x)
  return Dense(1, activation='sigmoid')(x)


def gru_comment(inpt1, inpt2):
  print('  exp    26 ')

  # First input (OHLC prices) using GRU layers
  x1 = GRU(50, return_sequences=True)(inpt1)
  x1 = Dropout(0.3)(x1)
  x1 = GRU(50, return_sequences=False)(x1)
  x1 = Dropout(0.3)(x1)

  # Second input (sentiment analysis) using TimeDistributed GRU layers
  x2 = TimeDistributed(GRU(1, return_sequences=False))(inpt2)
  x2 = GRU(7, return_sequences=True)(x2)
  x2 = Flatten()(x2)

  # Combine the two inputs
  x1 = Dense(7)(x1)  # Additional dense layer for the first input
  conc = Concatenate()([x1, x2])
  # Output layer (predicting stock price)
  return Dense(1, activation='sigmoid')(conc)


# **************************** FFNN *************************

def ffnn_nocomment(inpt):
  x = Dense(64, activation='relu')(inpt)
  x = Dense(64, activation='relu')(x)
  x = Flatten()(x)
  return Dense(1, activation='sigmoid')(x)


def ffnn_commnet(inpt1, inpt2):
  # First input (OHLC prices) with Dense layers
  x1 = Dense(64, activation='relu')(inpt1)
  x1 = Dense(32, activation='relu')(x1)
  x1 = Flatten()(x1)
  # Second input (sentiment analysis) with Dense layers
  x2 = Dense(64, activation='relu')(inpt2)
  x2 = Dense(32, activation='relu')(x2)
  x2 = Flatten()(x2)
  # Concatenate the outputs of both inputs
  conc = Concatenate()([x1, x2])
  
  # Output layer (predicting stock price)
  return Dense(1, activation='sigmoid')(conc)

# **************************** CNN-LSTM *************************


def cnn_lstm_nocomment(inpt):
  print('inpt shape in cnnlstm ', inpt.shape)
  x = Conv1D(filters=64, kernel_size=3, activation='relu')(inpt)
  x = MaxPooling1D(pool_size=2)(x)
  x = LSTM(50, return_sequences=False)(x)
  x = Dropout(0.2)(x)
  x = Flatten()(x)
  return Dense(1, activation='sigmoid')(x)


def cnn_lstm_comment(inpt1, inpt2):
  x1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inpt1)
  x1 = MaxPooling1D(pool_size=2)(x1)
  x1 = LSTM(50, return_sequences=False)(x1)
  x1 = Dropout(0.2)(x1)
  x1 = Flatten()(x1)

  # Second input (sentiment analysis) with LSTM layers
  x2 = TimeDistributed(LSTM(1, return_sequences=False))(inpt2)
  x2 = Flatten()(x2)

  # Concatenate the outputs of both inputs
  conc = Concatenate()([x1, x2])
  # Output layer (predicting stock price)
  return Dense(1, activation='sigmoid')(conc)

# **************************** Bi-LSTM *************************


def bi_lstm_nocomment(inpt):
  x = Bidirectional(LSTM(50, return_sequences=True))(inpt)
  x = Bidirectional(LSTM(50, return_sequences=False))(x)
  x = Dropout(0.2)(x)
  x = Flatten()(x)
  return Dense(1, activation='sigmoid')(x)


def bi_lstm_comment(inpt1, inpt2):
  # First input (OHLC prices) with Bidirectional LSTM layers
  x1 = Bidirectional(LSTM(50, return_sequences=False))(inpt1)
  x1 = Dropout(0.2)(x1)

  # Second input (sentiment analysis) with Bidirectional LSTM layers
  x2 = TimeDistributed(LSTM(1, return_sequences=False))(inpt2)
  print('x2 shape ', x2.shape)
  x2 = Flatten()(x2)

  # Concatenate the outputs of both inputs
  conc = Concatenate()([x1, x2])

  # Output layer (predicting stock price)
  return Dense(1, activation='sigmoid')(conc)
