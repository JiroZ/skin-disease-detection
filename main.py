import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
# from keras import Sequential, Input
# from keras.src.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
# from keras.utils import plot_model

# from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
# from keras.optimizers import Adam
# from keras.callbacks import ReduceLROnPlateau


from keras.metrics import Recall
from sklearn.metrics import classification_report, confusion_matrix

import itertools

import requests
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

print(tf.__version__)
print('Available Devices')
print(tf.config.list_physical_devices())

skinDataFrameCSV = pd.read_csv('input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')

print(skinDataFrameCSV.head())
print(skinDataFrameCSV.dtypes)
print(skinDataFrameCSV.describe())
print(skinDataFrameCSV.isnull().sum())

skinDataFrameCSV['age'].fillna(int(skinDataFrameCSV['age'].mean()), inplace=True)
print(skinDataFrameCSV.isnull().sum())

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
base_skin_dir = 'input/skin-cancer-mnist-ham10000/HAM10000_images/'

# Merge images from both folders into one dictionary

imageid_path_dict = {
    os.path.splitext(os.path.basename(x))[0]: x for x in
    glob(os.path.join(base_skin_dir, '*.jpg'))
}

print("Generated Path Model")

skinDataFrameCSV['path'] = skinDataFrameCSV['image_id'].map(imageid_path_dict.get)
skinDataFrameCSV['cell_type'] = skinDataFrameCSV['dx'].map(lesion_type_dict.get)
skinDataFrameCSV['cell_type_idx'] = pd.Categorical(skinDataFrameCSV['cell_type']).codes
skinDataFrameCSV.head()

print("Mapped Image Path Cells")
print('Loading Images...')

imageArray = []


def getImage(image):
    print(image)
    # skinDataFrameCSV['image'] = np.asarray(Image.open(image).resize((125, 100)))
    imageArray.append(np.asarray(Image.open(image).resize((125, 100))))


with ThreadPoolExecutor(200) as executor:
    executor.map(getImage, skinDataFrameCSV['path'])

skinDataFrameCSV['image'] = imageArray
# skinDataFrameCSV['image'] = skinDataFrameCSV['path'].map(lambda x: np.asarray(Image.open(x).resize((125, 100))))

print("Images loaded successfully")

n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize=(4 * n_samples, 3 * 7))
for n_axs, (type_name, type_rows) in zip(m_axs, skinDataFrameCSV.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)

# See the image size distribution - should just return one row (all images are uniform)
skinDataFrameCSV['image'].map(lambda x: x.shape).value_counts()

skinDataFrameCSV = skinDataFrameCSV[skinDataFrameCSV['age'] != 0]
skinDataFrameCSV = skinDataFrameCSV[skinDataFrameCSV['sex'] != 'unknown']

plt.figure(figsize=(20, 10))
plt.subplots_adjust(left=0.125, bottom=1, right=0.9, top=2, hspace=0.2)
plt.subplot(2, 4, 1)
plt.title("AGE", fontsize=15)
plt.ylabel("Count")
skinDataFrameCSV['age'].value_counts().plot.bar()

plt.subplot(2, 4, 2)
plt.title("GENDER", fontsize=15)
plt.ylabel("Count")
skinDataFrameCSV['sex'].value_counts().plot.bar()

plt.subplot(2, 4, 3)
plt.title("localization", fontsize=15)
plt.ylabel("Count")
plt.xticks(rotation=45)
skinDataFrameCSV['localization'].value_counts().plot.bar()

plt.subplot(2, 4, 4)
plt.title("CELL TYPE", fontsize=15)
plt.ylabel("Count")
skinDataFrameCSV['cell_type'].value_counts().plot.bar()

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
skinDataFrameCSV['dx'].value_counts().plot.pie(autopct="%1.1f%%")
plt.subplot(1, 2, 2)
skinDataFrameCSV['dx_type'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()

plt.figure(figsize=(25, 10))
plt.title('LOCALIZATION VS GENDER', fontsize=15)
sns.countplot(y='localization', hue='sex', data=skinDataFrameCSV)

plt.figure(figsize=(25, 10))
plt.title('LOCALIZATION VS CELL TYPE', fontsize=15)
sns.countplot(y='localization', hue='cell_type', data=skinDataFrameCSV)

plt.figure(figsize=(25, 10))
plt.subplot(131)
plt.title('AGE VS CELL TYPE', fontsize=15)
sns.countplot(y='age', hue='cell_type', data=skinDataFrameCSV)
plt.subplot(132)
plt.title('GENDER VS CELL TYPE', fontsize=15)
sns.countplot(y='sex', hue='cell_type', data=skinDataFrameCSV)

print("Applied Dataset Plots")

# Training model
features = skinDataFrameCSV.drop(columns=['cell_type_idx'], axis=1)
target = skinDataFrameCSV['cell_type_idx']
features.head()

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.25, random_state=666)
tf.unique(x_train_o.cell_type.values)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_test_mean) / x_test_std

# Perform one-hot encoding on the labels
y_train = to_categorical(y_train_o, num_classes=7)
y_test = to_categorical(y_test_o, num_classes=7)
print(y_test)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1, random_state=999)
# Reshape image in 3 dimensions (height = 100, width = 125 , canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(100, 125, 3))
x_test = x_test.reshape(x_test.shape[0], *(100, 125, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(100, 125, 3))

x_train = x_train.reshape(6696, 125 * 100 * 3)
x_test = x_test.reshape(2481, 125 * 100 * 3)
print(x_train.shape)
print(x_test.shape)

# define the keras model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim=37500))
model.add(tf.keras.layers.Dense(units=64, kernel_initializer='uniform', activation='relu'))
model.add(tf.keras.layers.Dense(units=64, kernel_initializer='uniform', activation='relu'))
model.add(tf.keras.layers.Dense(units=64, kernel_initializer='uniform', activation='relu'))
model.add(tf.keras.layers.Dense(units=7, kernel_initializer='uniform', activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00075,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-8)

# compile the keras model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# fit the keras model on the dataset
history = model.fit(x_train, y_train, batch_size=10, epochs=50)

accuracy = model.evaluate(x_test, y_test, verbose=1)[1]
print("Test: accuracy = ", accuracy * 100, "%")

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Set the CNN model
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*3 -> Flatten -> Dense*2 -> Dropout -> Out
input_shape = (100, 125, 3)
num_classes = 7

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.16))

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.20))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model.summary()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)

# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=4,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1, random_state=999)
# Reshape image in 3 dimensions (height = 100, width = 125 , canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(100, 125, 3))
x_test = x_test.reshape(x_test.shape[0], *(100, 125, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(100, 125, 3))
# With data augmentation to prevent overfitting

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.12,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.12,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

datagen.fit(x_train)

# Fit the model
epochs = 60
batch_size = 16

batch_x, batch_y = next(datagen.flow(x=x_train, y=y_train, batch_size=batch_size))

print(type(x_train))
print(type(y_train))

print(x_train.shape)
print(y_train.shape)

print(batch_x.shape)
print(batch_y.shape)

print(np.isnan(x_train).any())  # Check if x_train contains NaN values
print(np.isnan(y_train).any())  # Check if y_train contains NaN v

train_data_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

print(train_data_generator)
print(epochs)
print(x_train.shape[0] // batch_size)
print(1)
print([learning_rate_reduction])

history = model.fit(
    train_data_generator,
    epochs=epochs,
    steps_per_epoch=x_train.shape[0] // batch_size,
    validation_data=(x_validate, y_validate),
    verbose=1,
    callbacks=[learning_rate_reduction]
)

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

model.save("model.h5")

# # Function to plot confusion matrix
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# # Predict the values from the validation dataset
# Y_pred = model.predict(x_validate)
# # Convert predictions classes to one hot vectors
# Y_pred_classes = np.argmax(Y_pred, axis=1)
# # Convert validation observations to one hot vectors
# Y_true = np.argmax(y_validate, axis=1)
# # compute the confusion matrix
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
#
# # plot the confusion matrix
# plot_confusion_matrix(confusion_mtx, classes=range(7))
#
# # Predict the values from the validation dataset
# Y_pred = model.predict(x_test)
# # Convert predictions classes to one hot vectors
# Y_pred_classes = np.argmax(Y_pred, axis=1)
# # Convert validation observations to one hot vectors
# Y_true = np.argmax(y_test, axis=1)
# # compute the confusion matrix
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
#
# # plot the confusion matrix
# plot_confusion_matrix(confusion_mtx, classes=range(7))
#
# label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
# plt.bar(np.arange(7), label_frac_error)
# plt.xlabel('True Label')
# plt.ylabel('Fraction classified incorrectly')
