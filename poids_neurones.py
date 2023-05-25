import tensorflow as tf
import matplotlib.pyplot as plt

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model('model_.h5')

# Sélectionner la couche dont vous souhaitez visualiser les poids
layer_index = 0  # Indice de la couche souhaitée
layer = model.layers[layer_index]

# Obtenir les poids de la couche
weights = layer.get_weights()[0]

# Calculer l'importance des poids
importance = tf.reduce_sum(tf.abs(weights), axis=1).numpy()

# Normaliser les valeurs d'importance entre 0 et 1
importance_normalized = (importance - importance.min()) / (importance.max() - importance.min())

# Générer une carte de chaleur des poids des neurones
plt.imshow(importance_normalized.reshape(1, -1), cmap='hot', aspect='auto')
plt.colorbar()
plt.title("Carte de chaleur des poids des neurones")
plt.xlabel("Neurones")
plt.ylabel("Couche")
plt.show()

