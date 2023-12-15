import streamlit as st
import os
from keras.utils import image_dataset_from_directory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

#---------------------------------------------------------------#

# Function to load and preprocess datasets
def load_datasets():
    batch_size = 32
    image_size = (64, 64)
    validation_split = 0.2

    # Create the training dataset from the 'train' directory
    train_ds = image_dataset_from_directory(
        directory='images/training_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='training',
        seed=123
    )

    # Create the validation dataset from the 'train' directory
    validation_ds = image_dataset_from_directory(
        directory='images/training_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='validation',
        seed=123
    )

    # Create the testing dataset from the 'test' directory
    test_ds = image_dataset_from_directory(
        directory='images/test_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size
    )

    return train_ds, validation_ds, test_ds

#---------------------------------------------------------------#

# Function to create and train the model
def create_and_train_model(train_ds, validation_ds, epochs):
    NUM_CLASSES = 5
    IMG_SIZE = 128

    HEIGTH_FACTOR = 0.2
    WIDTH_FACTOR = 0.2

    # Create a sequential model with a list of layers
    model = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGTH_FACTOR, WIDTH_FACTOR),
        layers.RandomZoom(0.2),
        layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="sigmoid")
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs)

    return model, history

#---------------------------------------------------------------#

# Function to display plots
def display_plots(history):
    # Create a figure and a grid of subplots with a single call
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the loss curves on the first subplot
    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the figure
    st.pyplot(fig)
    
#---------------------------------------------------------------#

# Function to evaluate and display test accuracy
def display_test_accuracy(model, test_ds):
    test_loss, test_acc = model.evaluate(test_ds)
    st.write('Test accuracy:', test_acc)

# Main Streamlit app
def main():
    st.title("Image Classification Streamlit App")
    
    # Input field for the number of epochs
    epochs = st.number_input("Number of Epochs", min_value=1, value=20)

    # Button to trigger model building and training
    if st.button("Train Model"):
        # Load datasets
        train_ds, validation_ds, test_ds = load_datasets()

        # Display loading message
        with st.spinner("Training the model. Please wait..."):
            # Create and train the model
            model, history = create_and_train_model(train_ds, validation_ds, epochs)

        # Display training and validation curves
        st.subheader("Training and Validation Curves")
        display_plots(history)

        # Display test accuracy
        st.subheader("Test Accuracy")
        display_test_accuracy(model, test_ds)

if __name__ == "__main__":
    main()