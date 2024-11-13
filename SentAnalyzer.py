__author__ = 'A. Malekijoo'

"""

in this file we are going to fine-tune financial texts
including Eco dataset and our labeled dataset on
a pretrained model on BERT architecture.

"""

import re
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
print("tensorflow version ", tf.__version__)

# from tensorflow.python.keras.optimizer_v2.adam import Adam

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification

from scipy.special import softmax

nltk.download('stopwords')
nltk.download('wordnet')

# *****************************     just eco dataset     **************************************#

dfe = pd.read_csv('ecodata.csv', delimiter=',', encoding='latin-1', header=None)
dfe = dfe.rename(columns=lambda x: ['label', 'text'][x])
dfe.info()
dfe = dfe[['text', 'label']]
dfe['label'] = dfe['label'].map({'negative': 0.0, 'neutral': 1.0, 'positive': 2.0})



# *****************************    eco + comments dataset + weighted   **************************************#

# dfe = pd.read_csv('Eng_labeled_comments.csv')
# dfe.columns = ['index', 'username', 'name', 'text', 'label']
# dfe['label'] = dfe['label'].map({'negative': 0.0, 'neutral': 1.0, 'positive': 2.0})
# dfe = dfe[['label', 'text']]

# dfeco = pd.read_csv('ecodata.csv', delimiter=',', encoding='latin-1', header=None)
# dfeco = dfeco.rename(columns=lambda x: ['label', 'text'][x])
# dfeco['label'] = dfeco['label'].map({'negative': 0.0, 'neutral': 1.0, 'positive': 2.0})
# concat_df = pd.concat([dfeco, dfe], ignore_index=True)

concat_df = dfe

concat_df = concat_df.dropna(axis=0)
# class_counts = concat_df['label'].value_counts().to_dict()
# total = sum(class_counts.values())
# print(' \n shape and dist ', concat_df.shape, class_counts)
# weight_for_0 = (1 / class_counts[0]) * (total / 3.0)
# weight_for_2 = (1 / class_counts[2]) * (total / 3.0)
# weight_for_1 = (1 / class_counts[1]) * (total / 3.0)
# class_weights = {1.0: weight_for_1, 2.0: weight_for_2, 0.0: weight_for_0}
df = concat_df

# Function for text cleaning and preprocessing
def preprocess(text):

    # digit_exist = re.findall(r'\b\d+\b', text)
    # if bool(digit_exist):
    #   return np.nan
    
    # Convert to lowercase
    text = text.lower()
    
    tokens = []
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize the text
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        tokens.append(t)
    # tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # if len(tokens) < 7 or len(tokens) > 50:
    #   return np.nan
    # else:
    #   # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text


if __name__ == "__main__":
    
    #  models
    #  https://huggingface.co/models?pipeline_tag=text-classification&sort=trending&search=sentiment

    #  https://arxiv.org/abs/2202.03829
    #  https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

    #  Load the tokenizer and the pre-trained sentiment analysis model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

    #  Prepare your labeled dataset and split into train and validation sets
    train, val = train_test_split(df, test_size=0.1, random_state=37)

    # Clean and preprocess the text data
    train['text'] = train['text'].apply(preprocess)
    val['text'] = val['text'].apply(preprocess)
    
    train = train.dropna(axis=0, subset=['text'])
    val = val.dropna(axis=0, subset=['text'])

    # Tokenize the input datasets
    train_encodings = tokenizer(train['text'].tolist(), truncation=True, padding=True, max_length=2**15)
    val_encodings = tokenizer(val['text'].tolist(), truncation=True, padding=True, max_length=2**15)

    # Convert the labels to TensorFlow tensors
    train_labels = tf.convert_to_tensor(train['label'].tolist())
    val_labels = tf.convert_to_tensor(val['label'].tolist())

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

    # Fine-tune the model


    # optimizer = Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer='adam', loss=loss, metrics=[metric])
    # print(model.optimizer.get_config())
    # model.optimizer.learning_rate = 0.00001
    # model.optimizer.epsilon = 1e-08
    # model.optimizer.clipnorm = 1.
    # print(model.optimizer.get_config())
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               optimizer = 'adam',
    #               metrics=['accuracy'])

    print(' \n                    summary of model \n', model.summary())

    model.fit(train_dataset.shuffle(len(train_dataset)).batch(16), epochs=5, batch_size=16,
              validation_data=val_dataset.batch(16))

    # Save the fine-tuned model
    model.save_pretrained("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")


