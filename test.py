import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Charger le modèle pré-entraîné
model = load_model('model_.h5')
# Visualisation de l'architecture du modèle
plot_model(model, to_file='model_architecture.png', show_shapes=True)
