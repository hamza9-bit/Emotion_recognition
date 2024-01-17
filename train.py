# train.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from init_model import initialize_model

def train_model(data_path, epochs=10, test_size=0.2):
    model = initialize_model()

    datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=test_size)

    # Chargement des données et division en ensembles d'entraînement et de validation
    train_generator = datagen.flow_from_directory(data_path, target_size=(256, 256), batch_size=32, class_mode='categorical', subset='training')
    validation_generator = datagen.flow_from_directory(data_path, target_size=(256, 256), batch_size=32, class_mode='categorical', subset='validation')

    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Affichage des métriques
    print("Historique d'entraînement:", history.history)

    # Évaluation sur l'ensemble de test
    test_generator = datagen.flow_from_directory(data_path, target_size=(256, 256), batch_size=32, class_mode='categorical', subset='validation', shuffle=False)
    y_true = test_generator.classes

    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred_classes)

    # Affichage de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_generator.class_indices, yticklabels=train_generator.class_indices)
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.show()

    # Calcul de la précision
    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f'Précision sur l\'ensemble de test: {accuracy * 100:.2f}%')

    model.save("emotion_model.h5")
data_path="dataset"
# Entraînement du modèle
train_model(data_path, epochs=10)

