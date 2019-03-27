import cv2
import numpy as np
import tensorflow as tf

import tfdata


class CNN_Model(object):
    def __init__(self, mode, img, lr,
            dropout, epochs, batch_size,
            label_type, classes, log_dir, name):
        if mode == "Train" or mode == 'train':
            self.learning_rate = lr
            self.dropout = dropout
            self.epochs = epochs
            self.batch_size = batch_size
            self.classes = classes
            self.label_type = label_type
            self.log_dir = log_dir
            self.name = name
            self.load_data()
            self.train_model()
        elif mode == 'Test' or mode == 'test':
            self.name = name
            self.img = img
            self.test_model()


    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tfdata.load_data()

        print("X_train length = {}".format(len(x_train)))
        print("X_test length = {}".format(len(x_test)))
        print("Input shape = {}".format(x_train.shape))
        img_rows = x_train.shape[1]
        img_cols = x_train.shape[2]

        try:
            img_channels = x_train.shape[3]
        except Exception as e:
            img_channels = 1

        x_train = x_train.reshape(x_train.shape[0],
                x_train.shape[1], x_train.shape[2],
                img_channels) / 255
        x_test = x_test.reshape(x_test.shape[0],
                x_train.shape[1], x_train.shape[2],
                img_channels) / 255

        if self.label_type == 'one_hot':
            y_train = tf.keras.utils.to_categorical(y_train, self.classes)
            y_test = tf.keras.utils.to_categorical(y_test, self.classes)

        self.input_shape = (x_train.shape[1], x_train.shape[2], img_channels)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


    def init_model(self):
        self.model = tf.keras.models.Sequential(name=self.name)


    def loss(self):
        if self.label_type == 'one_hot':
            return tf.keras.losses.categorical_crossentropy
        else:
            return tf.keras.losses.sparse_categorical_crossentropy


    def optimizer(self):
        return tf.keras.optimizers.Adam()


    def callbacks(self):
        callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                    histogram_freq=2,
                    write_graph=True,
                    write_images=False),
                tf.keras.callbacks.ModelCheckpoint(
                    'test_dir/model/%s.h5'%self.name,
                    verbose=0,)
                ]
        return callbacks


    def compile_model(self):
        self.model.compile(loss=self.loss(),
                optimizer=self.optimizer(),
                metrics=['accuracy'])


    def fit_model(self):
        self.model.fit(self.x_train, self.y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                callbacks=self.callbacks(),
                validation_data=(self.x_test, self.y_test))


    def input_stem(self):
        self.model.add(tf.keras.layers.Conv2D(filters=32,
                kernel_size=(3,3),
                activation='relu',
                name='stem_cv1',
                input_shape=self.input_shape))
        self.model.add(tf.keras.layers.Conv2D(filters=32,
                kernel_size=(3,3),
                activation='relu',
                name='stem_cv2'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                name='stem_pool1'))
        self.model.add(tf.keras.layers.Dropout(self.dropout,
                name='stem_drop1'))


    def conv_layer(self, layer_count):
        for i in range(1, layer_count):
            self.model.add(tf.keras.layers.Conv2D(filters=64 * i,
                    kernel_size=(3,3),
                    activation='relu',
                    name='blck{}_cv{}'.format(i,i)))
            #this filter size is moved down to fix the neg dim error
            self.model.add(tf.keras.layers.Conv2D(filters=64 * i,
                    kernel_size=(3,3),
                    activation='relu',
                    name='blck{}_cv{}'.format(i,i+1)))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                    name='blck{}_pool{}'.format(i,i)))
            self.model.add(tf.keras.layers.Dropout(self.dropout,
                    name='blck{}_drop'.format(i)))



    def flatten(self):
        self.model.add(tf.keras.layers.Flatten(name='flatten'))


    def fully_connected(self, start_nodes, fc_count):
        for i in range(1, fc_count):
            self.model.add(tf.keras.layers.Dense(start_nodes / i,
                    activation='relu', name='fc{}'.format(i)))
            self.model.add(tf.keras.layers.Dropout(self.dropout,
                    name='fc{}_drop'.format(i)))


    def softmax(self):
        self.model.add(tf.keras.layers.Dense(self.classes,
            activation='softmax',
            name='predictions/softmax'))


    def train_model(self):
        self.init_model()
        self.input_stem()
        self.conv_layer(layer_count=3)
        self.flatten()
        self.fully_connected(start_nodes=256, fc_count=2)
        self.softmax()
        self.compile_model()
        self.fit_model()
        self.model.summary()
        tf.keras.models.save_model(self.model, self.name+'.h5')


    def test_model(self):
        loaded_model = tf.keras.models.load_model(self.name+'.h5')
        # this will fail if the image does not have channel defined
        img = cv2.imread(self.img)
        img = cv2.resize(img, (64, 64))
        img = img.reshape(1, 64, 64, 3) / 255
        prediction = loaded_model.predict(img)
        print("Prediction: \n %s" %prediction)
        print("Predicted_label: %s" %np.argmax(prediction[0]))



if __name__ == "__main__":
    model = CNN_Model(mode="train",
                img='test_img.jpg',
                lr=0.002,
                dropout=0.25,
                epochs=50,
                batch_size=128,
                classes=2,
                label_type="",
                log_dir='test_dir/logs',
                name='test_model_tinder')
