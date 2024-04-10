import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import flwr as fl

X_train = np.load('X_train1.npy')
y_train = np.load('y_train1.npy')
X_test = np.load('X_test1.npy')
y_test = np.load('y_test1.npy')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation and testing sets (only rescaling)
val_test_datagen = ImageDataGenerator()

# Train data generator
train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=32
)

# Validation data generator
val_generator = val_test_datagen.flow(
    X_val,
    y_val,
    batch_size=32
)

# Test data generator
test_generator = val_test_datagen.flow(
    X_test,
    y_test,
    batch_size=32,
    shuffle=False
)

# Load MobileNet model (pre-trained on ImageNet)
mobilenet_model = tf.keras.applications.MobileNet(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze pre-trained layers
for layer in mobilenet_model.layers:
    layer.trainable = False

# Add classification head
model = tf.keras.models.Sequential([
    mobilenet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.metrics_per_epoch = []

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        
        for epoch in range(5):
            history = model.fit(
                train_generator,
                steps_per_epoch=len(X_train) // 32,
                validation_data=val_generator,
                validation_steps=len(X_val) // 32
            )
            # Calculate precision, recall, and accuracy
            test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
            self.metrics_per_epoch.append({
                "precision": history.history['precision'][0],
                "recall": history.history['recall'][0],
                "accuracy": test_accuracy
            })
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // 32)
        return test_loss, len(X_test), {"accuracy": test_accuracy}
    
client = FlowerClient()
fl.client.start_numpy_client(server_address="127.0.0.1:5000", client=client)

# Access metrics per epoch
print(client.metrics_per_epoch)
