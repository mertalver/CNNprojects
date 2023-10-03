import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

dataset_train = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")

X = dataset_train.iloc[:, 1:].values
X = X / 255.0
y = dataset_train.iloc[:, 0].values

X_pred = dataset_test.iloc[:, :].values
X_pred = X_pred / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", input_shape=(28, 28, 1)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.3))
cnn.add(tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.3))
cnn.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.3))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=256, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=10, activation="softmax"))

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model = cnn.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose=1)

y_pred = cnn.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

y_test = lb.inverse_transform(y_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix)
disp.plot(cmap="OrRd_r")

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
plt.show()
