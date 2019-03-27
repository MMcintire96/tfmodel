import cv2
import numpy as np
import tensorflow as tf

import tfdata

model = tf.keras.models.load_model('test_model_lfw.h5')

(x_train, y_train), (x_test, y_test) = tfdata.load_data()

x_test = x_test / 255

print(y_test)

i = 0
for img in x_test:
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    preds = model.predict(img)
    labels = np.argmax(preds[0])
    print(preds)
    if labels == 1:
        print("predicted label HAPPY -- actual label %s" %y_test[i])
    elif labels == 0:
        print("predicted label NEUTRAL -- actual label %s" %y_test[i])
    else:
        print("ERRORS")
    i += 1
