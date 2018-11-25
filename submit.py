import ast
import datetime as dt
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import seaborn as sns
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from main import top_3_accuracy, INPUT_DIR, size, df_to_image_array_xd, preds2catids, list_all_categories, map3

# Force to run on CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

start = dt.datetime.now()

model_path = './model/weights-005-0.930.hdf5'

print('Loading model at', model_path)

# Load previous model
model = load_model(model_path, custom_objects={
                   'top_3_accuracy': top_3_accuracy})

print('Loaded model. Predicting')
test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
x_test = df_to_image_array_xd(test, size)
print(test.shape, x_test.shape)
print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3))


test_predictions = model.predict(x_test, batch_size=128, verbose=1)
top3 = preds2catids(test_predictions)

cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)

test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('gs_mn_submission_{}.csv'.format(
    int(map3 * 10**4)), index=False)

end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
