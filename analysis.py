# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# dimensions of our images.
img_width, img_height = 50, 50

train_data_dir = 'data/symlinks/train'
validation_data_dir = 'data/symlinks/validation'
test_data_dir = 'data/symlinks/test'
nb_epoch = 20
batch_size=128

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        samplewise_center=True,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
        samplewise_center=True,
        rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        follow_links=True)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        follow_links=True)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary',
        follow_links=True)

def get_model():
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

model = get_model()


## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
history = LossHistory()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'],
              verbose=1,
              shuffle=True)

model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        verbose=1,
        validation_steps=validation_generator.samples/batch_size,
        callbacks=[history])

test_loss, test_acc = model.evaluate_generator(
         test_generator,
         steps=test_generator.samples/batch_size)

print("test_loss: %.4f - test_acc: %.4f"%(test_loss, test_acc))

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()

plt.show()
