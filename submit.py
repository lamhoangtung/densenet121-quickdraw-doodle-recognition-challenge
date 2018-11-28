from common import *
import seaborn as sns
import pandas as pd
import numpy as np
from keras.models import Model, load_model
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
import datetime as dt
import ast
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


start = dt.datetime.now()

model_path = './model/3/weights-001-0.942.hdf5'

print('Loading model at', model_path)

# Load previous model
trained_model = load_model(model_path, custom_objects={
                   'top_3_accuracy': top_3_accuracy})

# TTA
model = TTA_ModelWrapper(trained_model)

print('Loaded model. Predicting')
test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))


max_load_step = int(test.shape[0] / LOAD_SIZE) + 1
#max_load_step = 4

cats = list_all_categories()

id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}

test_predictions = None


for load_step in range(max_load_step):
    x_test = df_to_image_array_xd(test, size, load_step)
    new_predictions = model.predict(x_test, batch_size=128, verbose=1)
    test_predictions = new_predictions if test_predictions is None else np.concatenate(
        (test_predictions, new_predictions))

top3 = preds2catids(test_predictions)
top3cats = top3.replace(id2cat)

valid_df = pd.read_csv(os.path.join(
    DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)))


map3 = mapk(valid_df[['y']].values, top3.values)
print('Map3: {:.3f}'.format(map3))


test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('gs_mn_submission_{}.csv'.format(
    int(map3 * 10**4)), index=False)


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
