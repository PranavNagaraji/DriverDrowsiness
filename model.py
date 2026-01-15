import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

import tensorflow as tf
early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)
#I've initiated early stopping here

# drowsy_dir='D:\\Stuff\\Data Science\\DDD\\EyeDataset\\drowsy'
# nondrowsy_dir='D:\\Stuff\\Data Science\\DDD\\EyeDataset\\nondrowsy'

dataset_dir="D:\\Stuff\\Data Science\\DDD\\train"

datagen=ImageDataGenerator(rescale=1.0/255.0)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'  # Since we're doing a binary classification (drowsy vs non-drowsy)
)

# Convolutional Newural Networks(CNN) are used primarily for image classification and computer vision tasks
# These learn automatically through layers
# Convolutional Layers: These help detect patterns like edges, textures or specific objects
    # Pooling Layers: These layers reduce the spatial dimensions (height and width) of the data while retaining important features. 
    # Common pooling methods are max-pooling and average pooling. 
    # Fully Connected Layers: After convolutional and pooling layers, the output is flattened and passed to fully connected layers,
        # which perform the actual classification based on learned features.
    # Activation Functions: Functions like ReLU (Rectified Linear Unit) introduce non-linearity to the model, 
        # allowing it to learn complex patterns.

# In our model:
# a) Conv2D Layer:
# * This layer performs convolution operations on the image.
# * The kernel (or filter) slides across the input image, computing dot products between the weights and input. 
#   These weights get adjusted during training to help recognize patterns.
# * **Activation function**: We use ReLU (Rectified Linear Unit) activation, which outputs the input if positive, 
#   otherwise zero. It introduces non-linearity, which is important for deep networks.

# b) MaxPooling2D Layer:
# * This layer helps reduce the dimensionality of the image and the number of parameters, 
#   making the network faster and less prone to overfitting.
# * It downsamples the image by taking the maximum value from a specific window (usually 2x2) of the feature map. 
#   This reduces the image's spatial size but retains the important features.

# c) Flatten Layer:
# * After convolution and pooling, we flatten the output to feed it into a fully connected layer. 
#   This converts the multi-dimensional feature maps into a 1D vector.

# d) Dense Layer:
# * Fully connected layers where each neuron is connected to every other neuron in the previous layer. 
#   These layers learn the final decision by combining the learned features.
# * The output layer has a **sigmoid activation function** because we're performing binary classification 
#   (drowsy or non-drowsy).

# This overall layer is sequantial

# Define the CNN model
model = models.Sequential([
    # First Convolutional Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    
    # 2nd CL
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # 3rd CL
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Flatten the output for the fully connected layer
    layers.Flatten(),
    
    # Fully Connected Layer
    layers.Dense(512, activation='relu'),
    
    # Output Layer (Binary classification: 1 or 0)
    layers.Dense(1, activation='sigmoid')
])

# Summary of the model
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Binary classification loss
    metrics=['accuracy']
)

history = model.fit(
    train_generator, 
    epochs=10,  # We can increase the epochs based on your need
    callbacks=early_stopping
)

# Model is successfully trained. Here's a quick summary of what the output confirms:
    # Data Loaded: 12,783 images from 2 classes (drowsy & non-drowsy).
    # Model Architecture: 3 convolutional layers, max-pooling, flattening, dense layers.
    # Training: Completed 10 epochs.
    # Accuracy: Reached ~99.1% — that's very good on training data, may lead to overfitting too!

model.save('model.keras')