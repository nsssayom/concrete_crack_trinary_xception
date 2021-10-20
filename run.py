# Imports
import imp
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import confusion_matrix
import os
import numpy as np
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Activation, Flatten, MaxPooling2D, SeparableConv2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Input, BatchNormalization, ReLU, ELU, Dropout, Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D, Concatenate
import seaborn as sns
from tensorflow.keras.utils import plot_model

img_rows, img_cols = 64, 64
train_data_dir = './dataset_new/train'
validation_data_dir = './dataset_new/val'


IMG_SHAPE = img_cols, img_rows, 3
classes = 3
batch_size = 64

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

datagen = ImageDataGenerator(rescale=1./255)

# automatically retrive images and their classes for training and validation
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_cols, img_rows),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')

num_train_samples = train_generator.samples
num_validation_samples = validation_generator.samples

print('num_train_samples: ' + str(num_train_samples))
print('num_validation_samples: ' + str(num_validation_samples))


def entry_flow(inputs):

    x = Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation = x

    for size in [128, 256, 728]:

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual = Conv2D(size, 1, strides=2, padding='same')(
            previous_block_activation)

        x = tensorflow.keras.layers.Add()([x, residual])
        previous_block_activation = x

    return x


def middle_flow(x, num_blocks=8):

    previous_block_activation = x

    for _ in range(num_blocks):

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        # x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        # x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        # x = BatchNormalization()(x)

        x = tensorflow.keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x

    return x


def exit_flow(x):

    previous_block_activation = x

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    #x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    #x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual = Conv2D(1024, 1, strides=2, padding='same')(
        previous_block_activation)
    x = tensorflow.keras.layers.Add()([x, residual])

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    #x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    #x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    return x


inputs = Input(shape=(img_rows, img_cols, 3))
outputs = exit_flow(middle_flow(entry_flow(inputs)))
xception = Model(inputs, outputs)

xception.summary()
# plot the model
plot_model(xception, to_file='./graph/xception.png')

opt = tensorflow.keras.optimizers.Adam()
xception.compile(loss='categorical_crossentropy',
                 optimizer=opt,
                 metrics=['accuracy'])


history_1 = xception.fit_generator(train_generator,
                                   steps_per_epoch=num_train_samples // batch_size,
                                   epochs=30,
                                   validation_data=validation_generator,
                                   validation_steps=num_validation_samples // batch_size)


#  "Accuracy"
plt.figure(figsize=[8, 4])
sns.set_theme()
plt.plot(history_1.history['accuracy'], color="blue")
plt.plot(history_1.history['val_accuracy'], color="red")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
# plt.show()
plt.gca().set_position([0.1, 0.12, 0.8, 0.8])
plt.savefig('./graph/TwoClass_accuracy_raw_1.svg')


# "Loss"
plt.figure(figsize=[8, 4])
sns.set_theme()
plt.plot(history_1.history['loss'], color="blue")
plt.plot(history_1.history['val_loss'], color="red")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()
plt.gca().set_position([0.1, 0.12, 0.8, 0.8])
plt.savefig('./graph/TwoClass_loss_raw_1.svg')
# plt.savefig('G:/Nazia/BanglaLekha/Comp-graph/Com_loss18.png')

%matplotlib inline

batch_size = 64

Y_pred = xception.predict_generator(
    validation_generator, num_validation_samples // batch_size+1)
print(Y_pred.shape)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

target_names = ['Cracked', 'Non-Cracked']
print(classification_report(y_true, y_pred, target_names=target_names, digits=3))

cm = confusion_matrix(y_true, y_pred)

classes = 3


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1]), ):
        plt.text(j, i, cm[i, j, ],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


imp.reload(sns)

sns.reset_defaults()

cm_plot_labels = ['Cracked', 'Non-Cracked']
# sns.set_theme()
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
plt.savefig('./graph/TwoClass_conf_Raw_1.svg')
# plt.savefig('G:/Nazia/BanglaLekha/Comp-graph/cf_val_git_05.png')
