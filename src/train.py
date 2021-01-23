from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parent_path = os.path.dirname(__file__)
src_path = os.path.dirname(parent_path)
sys.path.append(parent_path)
sys.path.append(src_path)


class CustomModel:
    def __init__(self, train_dir, val_dir, test_dir):
        self.batchsize = 48
        self.img_height = 48  # 224
        self.img_width = 48  # 224
        self.channels = 3

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        self.train_datagen = None
        self.val_datagen = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

    def import_augmented_data(self):
        """data augmentation on train/val sets"""
        self.train_datagen = ImageDataGenerator(rotation_range=20.0,
                                                horizontal_flip=True,
                                                shear_range=0.1,
                                                rescale=1./255)

        self.val_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(self.train_dir,
                                                                      target_size=(self.img_height, self.img_width),
                                                                      batch_size= self.batchsize,
                                                                      class_mode='categorical',
                                                                      shuffle=True)

        self.validation_generator = self.val_datagen.flow_from_directory(self.val_dir,
                                                                          target_size=(self.img_height, self.img_width),
                                                                          batch_size=self.batchsize,
                                                                          class_mode='categorical')

        self.test_generator = self.val_datagen.flow_from_directory(self.test_dir,
                                                                target_size=(self.img_height, self.img_width),
                                                                batch_size=self.batchsize,
                                                                class_mode='categorical')


    def create_model(self):
        """https://github.com/atulapra/Emotion-detection/blob/master/src/emotions.py"""

        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(3, activation='softmax'))

        # compile our model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=0.0003, decay=1e-6),
                           metrics=['accuracy'])

        return self.model.summary()

    def train_model(self, model_output_path, epochs=1):
        """ commence model training"""

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            patience=3,
            min_delta=0.00001)

        checkpoint = ModelCheckpoint(
            filepath=model_output_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max')

        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=1,
            verbose=1,
            factor=0.2,
            min_lr=0.00001)

        self.model.fit(self.train_generator,
                       steps_per_epoch=self.train_generator.samples // self.batchsize,
                       validation_data=self.validation_generator,
                       validation_steps=self.validation_generator.samples // self.batchsize,
                       epochs=epochs,
                       callbacks=[early_stopping, checkpoint, learning_rate_reduction])

    def evaluate_model(self):
        """compute test accuracy"""

        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        return test_loss, test_accuracy


if __name__ == '__main__':

    # define the directories
    train_dir = os.path.join('images', 'train')
    val_dir = os.path.join('images', 'valid')
    test_dir = os.path.join('images', 'test')

    # define the model
    model = CustomModel(train_dir, val_dir, test_dir)
    model.import_augmented_data()
    model.create_model()

    # train model
    model.train_model('emotions.h5', epochs=20)

    test_loss, test_accuracy = model.evaluate_model()

    logger.info(f"the test loss is {test_loss}")
    logger.info(f"the test acc is {test_accuracy}")

