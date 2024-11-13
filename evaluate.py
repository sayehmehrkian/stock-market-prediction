from datetime import datetime
import numpy as np
import pandas as pd
import re
import nltk
import tensorflow as tf

from sklearn import preprocessing
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from scipy.special import softmax
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoConfig
from transformers import TFAutoModelForSequenceClassification

nltk.download('stopwords')
nltk.download('wordnet')

pred_list = []
correctly_classified = []
incorrectly_classified = []


def performance_report(cm):
    col = len(cm)

    # col=number of class
    arr = []
    for key, value in cm.items():
        arr.append(value)

    cr = dict()
    support_sum = 0

    # macro avg of support is
    # sum of support only, not the mean.
    macro = [0] * 3

    # weighted avg of support is
    # sum of support only, not the mean.
    weighted = [0] * 3
    for i in range(col):
        vertical_sum = sum([arr[j][i] for j in range(col)])
        horizontal_sum = sum(arr[i])
        p = arr[i][i] / vertical_sum
        r = arr[i][i] / horizontal_sum
        f = (2 * p * r) / (p + r)
        s = horizontal_sum
        row = [p, r, f, s]
        support_sum += s
        for j in range(3):
            macro[j] += row[j]
            weighted[j] += row[j] * s
        cr[i] = row

        # add Accuracy parameters.
    truepos = 0
    total = 0
    for i in range(col):
        truepos += arr[i][i]
        total += sum(arr[i])

    cr['Accuracy'] = ["", "", truepos / total, support_sum]

    # Add macro-weight and weighted_avg features.
    macro_avg = [Sum / col for Sum in macro]
    macro_avg.append(support_sum)
    cr['Macro_avg'] = macro_avg

    weighted_avg = [Sum / support_sum for Sum in weighted]
    weighted_avg.append(support_sum)
    cr['Weighted_avg'] = weighted_avg

    print("Performance report of the model is :")
    space, p, r, f, s = " ", "Precision", "Recall", "F1-Score", "Support"
    print("%13s %9s %9s %9s %9s\n" % (space, p, r, f, s))
    stop = 0
    for key, value in cr.items():
        if stop < col:
            stop += 1
            print("%13s %9.2f %9.2f %9.2f %9d" % (key, value[0],
                                                  value[1],
                                                  value[2],
                                                  value[3]))
        elif stop == col:
            stop += 1
            print("\n%13s %9s %9s %9.2f %9d" % (key, value[0],
                                                value[1],
                                                value[2],
                                                value[3]))
        else:
            print("%13s %9.2f %9.2f %9.2f %9d" % (key,
                                                  value[0],
                                                  value[1],
                                                  value[2],
                                                  value[3]))


# Function for confusion matrix
def Conf_matrix(y_test, y_pred, target_names=None):
    # target_names is a list.
    # actual values are arranged in the rows.
    # predicted values are arranged in the columns.
    # if there are m classes, then cm is m*m matrix.
    if target_names == None:
        m = len(set(y_test))
    else:
        m = len(target_names)
    size = len(y_test)
    matrix = dict()

    # create matrix initialised with 0
    for class_name in range(m):
        matrix[class_name] = [0 for k in range(m)]

        # populating the matrix.
    for i in range(size):
        actual_class = y_test[i]
        pred_class = y_pred[i]
        matrix[actual_class][pred_class] += 1

    # Change the name of columns.
    if target_names == None:
        # Now, lets print the confusion matrix.
        print("Confusion Matrix of given model is :")
        if m == 3:
            print("Count=%-14d %-15s %-15s %-15s" % (size,
                                                     '0', '1',
                                                     '2'))
            for key, value in matrix.items():
                print("Actual %-13s %-15d %-15d %-15d" %
                      (key, value[0], value[1], value[2]))
        elif m == 2:
            print("Count=%-14d %-15s %-15s" % (size, '0', '1'))
            for key, value in matrix.items():
                print("Actual %-13s %-15d %-15d" % (key, value[0],
                                                    value[1]))
    else:
        matrix = dict(zip(target_names, list(matrix.values())))

        # Now, lets print the confusion matrix.
        print("Confusion Matrix of given model is :")
        print("Count=%-14d %-15s %-15s %-15s" %
              (size, target_names[0], target_names[1], target_names[2]))
        for key, value in matrix.items():
            print("Actual %-13s %-15d %-15d %-15d" %
                  (key, value[0], value[1], value[2]))

    return matrix


# Function for text cleaning and preprocessing
def preprocess(text):
    text = str(text)

    # if digit is in the string, it skip the comment
    # digit_exist = re.findall(r'\b\d+\b', text)
    # if bool(digit_exist):
    #     return np.nan
    # print('Raw text:    ', text)
    tokens = []

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # print('remove special characters:   ', text)

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

    # filter
    # if len(tokens) < 7 or len(tokens) > 40:
    #     return np.nan
    # else:
        # Join the tokens back into a single string
    text = ' '.join(tokens)
    return text


#  Prediction Function
def predict(text):
    # Tokenize the input text
    # text = str(text)
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="tf")

    # Pass the input through the model
    outputs = model(inputs)
    # Get the predicted sentiment scores
    predicted_scores = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    # Get the predicted sentiment label
    predicted_label = tf.argmax(predicted_scores).numpy()

    scores = outputs[0][0].numpy()
    scores = softmax(scores)
    # scores = softmax(outputs[0].numpy())
    return predicted_label

start_dt = datetime.now()
# Load the tokenizer and the fine-tuned model
model_add = './fine-tuned-model'
tokenizer = AutoTokenizer.from_pretrained(model_add)
config = AutoConfig.from_pretrained(model_add)
model = TFAutoModelForSequenceClassification.from_pretrained(model_add)

dfe = pd.read_csv('ecodata.csv', delimiter=',', encoding='latin-1', header=None)
dfe = dfe.rename(columns=lambda x: ['label', 'text'][x])
dfe.info()
dfe = dfe[['text', 'label']]
dfe['label'] = dfe['label'].map({'negative': 0.0, 'neutral': 1.0, 'positive': 2.0})

dfe = dfe.dropna(axis=0)


#  Prepare your labeled dataset and split into train and validation sets
train, val = train_test_split(dfe, test_size=0.1, random_state=37)
print('shape of train and val ', dfe.shape, train.shape, val.shape)
# Clean and preprocess the text data
print(val['label'])
val['text'] = val['text'].apply(preprocess)
print('after preproocess', val.shape)
val = val.dropna(axis=0, subset=['text'])
print('after drop ', val.shape)

# Tokenize the input datasets
# val_encodings   = tokenizer(val['text'].tolist(), truncation=True, padding=True)

# Convert the labels to TensorFlow tensors
# val_labels  = tf.convert_to_tensor(val['label'].tolist())

# val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

val['pred'] = val['text'].apply(predict)


label = val['label'].tolist()
pred  = val['pred'].tolist()

print('len ', len(label), len(pred))


index = 0
for actual, predict in zip(label, pred):
    if actual == predict:
        correctly_classified.append(index)
    else:
        incorrectly_classified.append(index)
    index += 1

ccc = len(correctly_classified)
icc = len(incorrectly_classified)
print('Correctly classified items  : {:5d} ({:=5.2f} %)'.format(ccc, ccc * 100 / (ccc + icc)))
print('Incorrectly classified items: {:5d} ({:=5.2f} %)'.format(icc, icc * 100 / (ccc + icc)))

# Calculate confusion matrix
conf_matrix = confusion_matrix(label, pred)
print(conf_matrix)
# conf_matrix=performance_report(conf_matrix)

# classes=['negative', 'neutral', 'positive']
classes = [0, 1, 2]
cm=Conf_matrix(label, pred, classes)
cr=performance_report(cm)


# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [False, True])
# # cm_display = np.array(cm_display)

print(datetime.now() - start_dt)

