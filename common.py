#!/usr/bin/env python
# coding: utf-8

# In[3]:


import ast
import os
import cv2
import numpy as np
import pandas as pd

from tensorflow import keras
from keras.applications.densenet import preprocess_input
from keras.metrics import (categorical_accuracy, top_k_categorical_accuracy)
from keras.models import Model, load_model


# In[10]:


DP_DIR = './kaggle/shuffle-csvs/'
INPUT_DIR = './kaggle/'

BASE_SIZE = 256
NCSVS = 100
NCATS = 340

batchsize = 330
size = 64
STEPS = 10000
EPOCHS = 27

LOAD_SIZE = batchsize * 100

# EPOCHS = trunc(30060000/(batchsize*STEPS))
# STEPS = trunc((34000000/EPOCHS)/batchsize)


# In[4]:


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


# In[46]:


def df_to_image_array_xd(df, size, load_step=0, lw=6, time_color=True):
    upper = min(LOAD_SIZE * (load_step + 1), df.shape[0])
    lower = LOAD_SIZE * load_step
    df_drawing = df['drawing'][lower:upper].apply(ast.literal_eval)
    x = np.zeros((len(df_drawing), size, size, 3))
    for i, raw_strokes in enumerate(df_drawing.values):
        x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x


# In[40]:


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])


# In[41]:


def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)


# In[42]:


def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img


# In[43]:


def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# In[1]:


def f2cat(filename: str) -> str:
    return filename.split('.')[0]


# In[ ]:




