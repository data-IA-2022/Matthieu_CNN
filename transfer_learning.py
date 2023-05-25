import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

SIZE = 64
BATCH_SIZE = 15
SEED = 32

with open("dataset_transfer" + str(SIZE) + ".pickle", "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame.from_dict(data)

train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['labels'])

le = LabelEncoder()

train_images = np.array(train_df['images'].to_list())  # / 255.0
train_labels = le.fit_transform(np.array(train_df['labels'].to_list()))

test_images = np.array(test_df['images'].to_list())  # / 255.0
test_labels = le.fit_transform(np.array(test_df['labels'].to_list()))

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(SIZE, SIZE, 3))

x = Flatten()(base_model.output)
output = Dense(12, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.summary()

model.compile(optimizer="adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_images,
                    train_labels,
                    validation_split=0.1,
                    epochs=20,
                    batch_size=BATCH_SIZE)

history.history.keys()

with open('history.pickle', 'wb') as f:
    pickle.dump(history.history, f)

model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)

model.save("modele_transfer_learning.h5")
