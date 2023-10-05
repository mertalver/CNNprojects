import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import img_to_array, load_img
from sklearn.model_selection import train_test_split

folder = "UTKFace"

age = []
gender = []
image_path = []

for file in os.listdir(folder):
    age.append(int(file.split('_')[0]))
    gender.append(int(file.split('_')[1]))
    image_path.append(os.path.join(folder, file))

dataset = pd.DataFrame({'Age': age, 'Gender': gender, 'Image': image_path})
dataset = dataset.sample(frac=1, random_state=42)
dataset = dataset.reset_index(drop=True)

X = np.array([img_to_array(load_img(img_path, target_size=(64, 64))) for img_path in dataset['Image']]) / 255.0
y = dataset.drop("Image", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

age_model = tf.keras.models.Sequential()
age_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(64, 64, 3)))
age_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
age_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
age_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
age_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"))
age_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
age_model.add(tf.keras.layers.Flatten())
age_model.add(tf.keras.layers.Dense(128, activation="relu"))
age_model.add(tf.keras.layers.Dense(64, activation='relu'))
age_model.add(tf.keras.layers.Dense(32, activation='relu'))
age_model.add(tf.keras.layers.Dense(1, activation="linear"))

gender_model = tf.keras.models.Sequential()
gender_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(64, 64, 3)))
gender_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
gender_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
gender_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
gender_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"))
gender_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
gender_model.add(tf.keras.layers.Flatten())
gender_model.add(tf.keras.layers.Dense(128, activation="relu"))
gender_model.add(tf.keras.layers.Dense(64, activation="relu"))
gender_model.add(tf.keras.layers.Dense(32, activation="relu"))
gender_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

input_layer = tf.keras.layers.Input(shape=(64, 64, 3))
age_output = age_model(input_layer)
gender_output = gender_model(input_layer)

combined_model = tf.keras.models.Model(inputs=input_layer, outputs=[age_output, gender_output])
combined_model.compile(optimizer="adam", loss=["mean_squared_error", "binary_crossentropy"], metrics=["mae", "accuracy"])
combined_model.fit(X_train, [y_train["Age"], y_train["Gender"]], epochs=25, batch_size=32, validation_data=(X_test, [y_test["Age"], y_test["Gender"]]))

pred_image = load_img("2.jpg", target_size=(64, 64))
pred_image = np.expand_dims(pred_image, axis=0)
y_pred = combined_model.predict(pred_image/255.0)
predicted_age = y_pred[0][0]
predicted_gender = "Female" if y_pred[1][0] > 0.5 else "Male"

print("Predicted Age:", round(predicted_age[0]))
print("Predicted Gender:", predicted_gender)

