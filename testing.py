import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model=load_model('model.keras')

img_folder="D:\\Stuff\\Data Science\\DDD\\training_dataset\\Open_Eyes"

def predict_image(img_path):
    img=image.load_img(img_path, target_size=(64,64))
    img_array=image.img_to_array(img) 
    img_array=img_array/255.0 #normalize the array
    img_array=np.expand_dims(img_array, axis=0)

    # We have an image that has the shape (64, 64, 3), which represents
    # a single image with 64x64 pixels and 3 color channels (RGB).

    # However, Keras models expect input to have a batch dimension.
    # Even if you're predicting just one image, the model still expects 
    # it in the format of (batch_size, height, width, channels).

    # np.expand_dims(img_array, axis=0) adds a batch dimension to the image.
    # It converts the shape from (64, 64, 3) to (1, 64, 64, 3),
    # where '1' is the batch size, indicating that we are passing one image.

    # The reason for this is that Keras models expect the input to be a batch of data,
    # even if it's just a single image. This ensures consistency with how data was 
    # handled during training, where the model was trained on batches of images.

    # So, after this line, `img_array` becomes a batch of 1 image with shape (1, 64, 64, 3),
    # which can now be passed into the model for prediction.

    prediction=model.predict(img_array)[0][0]

    if prediction<0.5:
        print('Non-Drowsy')
    else:
        print('Drowsy')

for img in os.listdir(img_folder):
    img_path=os.path.join(img_folder, img)
    predict_image(img_path)