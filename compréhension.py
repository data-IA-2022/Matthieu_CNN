import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tf_explain.core.grad_cam import GradCAM
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

# Charger votre modèle entraîné
model = tf.keras.models.load_model('model_.h5')

# Charger l'image à analyser
img_path = '/home/matthieu/Formation_IA/Briefs/deep_learning_projet1/kagglecatsanddogs_5340/PetImages/Dog/1002.jpg'
img = image.load_img(img_path, target_size=(224, 224))

# Convertir l'image en un tableau numpy
x = image.img_to_array(img)

# Ajouter une dimension supplémentaire pour représenter le lot de données (batch)
x = np.expand_dims(x, axis=0)

# Prétraiter l'image de la même manière que lors de l'entraînement de votre modèle
x = preprocess_input(x)

# Obtenir les prédictions pour l'image
predictions = model.predict(x)

# Afficher les probabilités pour chaque classe
print(predictions)


# Initialiser GradCAM avec le modèle et la couche de sortie souhaitée
explainer = GradCAM()

# Obtenir le gradient des activations pour la classe prédite
grid = explainer.explain((x, None), model, class_index=np.argmax(predictions))

# Afficher la heatmap avec les régions les plus importantes pour la prédiction
plt.imshow(grid, cmap='jet')
plt.show()
