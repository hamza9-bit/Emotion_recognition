# main.py
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image  # Import Image module from Pillow
import os


def test_model(img_path):
    img = Image.open(img_path).convert('L')  # Use Image module from Pillow
    img = img.resize((48, 48))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    model = load_model("model/emotion_model.h5") # Utilisez la fonction d'initialisation du modèle
   

      # Faites une prédiction
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    dataset_path = "dataset"
    class_names = sorted(os.listdir(dataset_path))
    
    # Obtenez le nom de la classe à partir de l'index
    predicted_class_name = class_names[predicted_class_index]
    
    print("Prédiction de classe:", predicted_class_name)
    return predicted_class_name



# # Test du modèle
# test_image_path = "test_img.png"
# test_model(test_image_path)
