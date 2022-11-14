# A simple classifier using keras for the mask dataset that calculates the model accuracy.

# This code expects images to be in two folders: 
#   ./data/with_mask/ and ./data/without_mask/

model_num = 2 # select a model to test, valid values are 0-2
batch_size = 32
epochs = 1
image_size = (224, 224)
validation_split = 0.2

# Based on https://keras.io/getting_started/

import tensorflow as tf
from tensorflow import keras
from keras import layers
import time

classes = ("with_mask", "without_mask")

print("Loading data...")

seed = int(time.time())

# https://keras.io/api/preprocessing/image/
training_dataset = keras.preprocessing.image_dataset_from_directory(
  "./data/",
  labels="inferred", # keras bases the labels on the directories of the images
  label_mode="categorical", # this means that the label is either [1, 0] or [0, 1]
  batch_size=batch_size,
  image_size=image_size,
  validation_split=validation_split,
  subset="training",
  seed=seed)

validation_dataset = keras.preprocessing.image_dataset_from_directory(
  "./data/", 
  labels="inferred",
  label_mode="categorical",
  batch_size=batch_size,
  image_size=image_size,
  validation_split=validation_split,
  subset="validation",
  seed=seed)

def label_to_string(label): # convert a vector label to a descriptive string
  if(label[0]):
    return classes[0]
  else:
    return classes[1]

# Show a random example and print its label
def show_random_example():
  import matplotlib.pyplot as plt

  for images, labels in training_dataset:
    scaled_images = keras.layers.Rescaling(scale=1./255.)(images)
    plt.imshow(scaled_images[0])
    print(label_to_string(labels[0]))
    print(images[0].shape)
    break # only show one example

#Based on https://keras.io/getting_started/intro_to_keras_for_engineers/

print("Creating model...")

input_shape = (image_size[0], image_size[1], 3)

def create_model(model_num):
  if(model_num == 0):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs)
  elif(model_num == 1):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs)
  elif(model_num == 2):
    # From training.ipynb (rnale1)
    baseModel = keras.applications.MobileNetV2(weights="imagenet", include_top=False, 
      input_tensor=keras.Input(shape=input_shape))
    headModel = baseModel.output
    headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = layers.Flatten(name="flatten")(headModel)
    headModel = layers.Dense(128, activation="relu")(headModel)
    headModel = layers.Dropout(0.5)(headModel)
    headModel = layers.Dense(2, activation="softmax")(headModel)
    return keras.Model(inputs=baseModel.input, outputs=headModel)


model = create_model(model_num)

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
              loss=keras.losses.CategoricalCrossentropy())

print("Training model...")

model.fit(training_dataset, epochs=epochs, verbose=1)

# Test validation set and count correct predictions one batch at a time
print("Testing model accuracy...")

total_examples = 0
correct_examples = 0

for images, labels in validation_dataset:
  predictions = model(images)
  for i, label in enumerate(labels):
    total_examples += 1
    prediction = predictions[i]
    if prediction[0] > prediction[1] and label[0] == 1:
      correct_examples += 1
    elif prediction[1] > prediction[0] and label[1] == 1:
      correct_examples += 1
print(str(correct_examples) + "/" + str(total_examples) + " correct")
print("accuracy="+str(round(correct_examples/total_examples*100., 2))+"%")

print("Evaluating model...")

metric = model.evaluate(validation_dataset)
print(model.metrics_names)
print(metric)
