import ast
import os
import matplotlib.pyplot as plt

import cv2
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input

import keras
import pandas as pd

INPUT_SHUFFLED_DIR = '/mnt/raid1/kaggle/shuffle-csvs/'
NUMBERS_OF_CLASSES = 340
IMAGE_SIZE = 128
VALIDATION_SIZE = 5000


def draw_image(raw_strokes, size=IMAGE_SIZE, line_width=6, time_color=True):
    img = np.zeros((size, size), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0])-1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, line_width)
    return img


def image_generator(batch_size, ks):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(
                INPUT_SHUFFLED_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batch_size):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), IMAGE_SIZE, IMAGE_SIZE, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_image(raw_strokes)
                # x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(
                    df.y, num_classes=NUMBERS_OF_CLASSES)
                yield x, y


def df_to_image_array(df):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), IMAGE_SIZE, IMAGE_SIZE, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_image(raw_strokes)
    # x = preprocess_input(x).astype(np.float32)
    return x


valid_df = pd.read_csv(os.path.join(
    INPUT_SHUFFLED_DIR, 'train_k{}.csv.gz'.format(99)), nrows=VALIDATION_SIZE)
x_valid = df_to_image_array(valid_df)
y_valid = keras.utils.to_categorical(
    valid_df.y, num_classes=NUMBERS_OF_CLASSES)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3))

train_datagen = image_generator(batch_size=100, ks=range(99))

x, y = next(train_datagen)
n = 8
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    (-x[i]+1)/2
    ax.imshow((-x[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
    ax.axis('off')
plt.tight_layout()
fig.savefig('gs.png', dpi=300)
