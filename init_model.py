# model_init.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def initialize_model():
    model = Sequential()
    
    # Couche convolutive 1
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Couche convolutive 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Couche convolutive 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Couche flattening
    model.add(Flatten())

    # Couche fully connected 1
    model.add(Dense(128, activation='relu'))

    # Couche de sortie
    model.add(Dense(3, activation='softmax'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
