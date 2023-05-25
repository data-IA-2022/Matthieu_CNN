import tensorflow as tf

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model('model_.h5')

# Sélectionner la couche dont vous souhaitez obtenir les noms des neurones
layer_index = 0  # Indice de la couche souhaitée
layer = model.layers[layer_index]

# Vérifier la taille des neurones dans la couche
num_neurons = layer.get_weights()[0].shape[1]

if num_neurons > 0:
    # Obtenir les poids de la couche
    weights = layer.get_weights()[0]

    # Calculer la somme des valeurs absolues des poids par neurone
    neuron_sums = tf.reduce_sum(tf.abs(weights), axis=0).numpy()

    # Obtenir les indices des neurones triés par importance décroissante
    sorted_indices = tf.argsort(neuron_sums, direction='DESCENDING').numpy()

    # Obtenir les noms des neurones dans la couche
    neuron_names = layer.get_config()['name']

    # Afficher les noms des neurones ayant les poids les plus élevés
    num_neurons_to_display = min(10, num_neurons)  # Nombre de neurones à afficher
    for i in range(num_neurons_to_display):
        neuron_index = sorted_indices[i]
        neuron_name = neuron_names + '_' + str(neuron_index)
        print(f"Neurone {neuron_name} a un poids élevé.")
else:
    print("La couche sélectionnée n'a pas de neurones.")
