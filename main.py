# Import the necessary libraries and a few helper functions.
import ast
import datetime as dt
import os
import time
from math import trunc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           top_k_categorical_accuracy)
from keras.models import Model, load_model
from keras.optimizers import Adam
from tensorflow import keras

from common import *

start = dt.datetime.now()


"""
DenseNet 121
DenseNet architecture is new, it is a logical extension of ResNet. ResNet architecture has a fundamental building block (Identity)
where you merge (additive) a previous layer into a future layer. Reasoning here is by adding additive merges we are forcing the network 
to learn residuals (errors i.e. diff between some previous layer and current one). In contrast, DenseNet paper proposes concatenating 
outputs from the previous layers instead of using the summation.

https://arxiv.org/abs/1608.06993
"""


batchsize = 330
size = 64
STEPS = 10000

# You can set as many epoch as you want until don't see any improvement
EPOCHS = 45

base_model = DenseNet121(include_top=False, weights='imagenet',
                         input_shape=(size, size, 3), classes=NCATS)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NCATS, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4, decay=1e-9), loss='categorical_crossentropy', metrics=[
              categorical_crossentropy, categorical_accuracy, top_3_accuracy])

# Load previous checkpoint
# model = load_model('path_to_checkpoint', custom_objects={'top_3_accuracy': top_3_accuracy})

print(model.summary())

"""
Training with Image Generator
Keep in mind that Keras Densenet only accept RGB images, for this I only fill 3 channel with the same grayscale value 
so the image is not RGB, but it still works. You should try to enconde more information at this part, it will definitely 
help increase the LB accuracy

"""

valid_df = pd.read_csv(os.path.join(
    DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)))
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3))

train_datagen = image_generator_xd(
    size=size, batchsize=batchsize, ks=range(NCSVS - 1))

x, y = next(train_datagen)
n = 8
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True,
                        sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    (-x[i]+1)/2
    ax.imshow((-x[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
    ax.axis('off')
plt.tight_layout()
fig.savefig('gs.png', dpi=300)

# I used Tensorboard for learning visualization
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                          batch_size=batchsize, write_images=True, update_freq=(STEPS/10)*batchsize)
weightpath = "/mnt/raid1/kaggle/model/weights-{epoch:03d}-{top_3_accuracy:.3f}.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_loss', verbose=0,
                             save_best_only=False, save_weights_only=False, mode='auto', period=1)


model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks=[tensorboard, checkpoint]
)


valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.3f}'.format(map3))

# Create Submission

test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
test.head()
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
