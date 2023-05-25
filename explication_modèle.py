import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob

def explication_modèle():
    tab1, tab2,tab3 ,tab5 = st.tabs(["Comprendre l'architecture du modèle",
                                     "Heatmap des erreurs",
                                     "Avoir les poids des modèles",              
                                                    "GradCam"])
    
    with tab1:
        st.title("Visualiser l'architecture")
        st.write('Le code')
        st.image('/home/matthieu/Images/Captures d’écran/Capture d’écran du 2023-05-25 11-50-41.png')
        st.write('Le résultat ')
        st.image('model_architecture.png')

        
        
        st.title("Comprendre les différentes couches")
        st.write(" Chaque type de couche a un rôle spécifique dans le traitement des données :")
        st.write("-Les couches de convolution sont responsables de l'extraction des caractéristiques de l'image en appliquant des filtres convolutifs.")
        st.write("-Les couches de pooling réduisent la taille spatiale des caractéristiques extraites et permettent une meilleure généralisation.")
        st.write("-Les couches fully connected sont des couches de neurones traditionnelles où chaque neurone est connecté à tous les neurones de la couche précédente.")
        
    with tab2:
        st.title("Faire une heatmap de vos faux positifs(travail en cours)")
        
        """tf.config.run_functions_eagerly(True)
        model = load_model('model_.h5')
        
        

        # Dossier contenant les images de chats
        cat_images_folder = "/home/matthieu/Formation_IA/Briefs/deep_learning_projet1/kagglecatsanddogs_5340/PetImages/Cat"

        # Dossier contenant les images de chiens
        dog_images_folder = "/home/matthieu/Formation_IA/Briefs/deep_learning_projet1/kagglecatsanddogs_5340/PetImages/Dog"

        # Liste des chemins d'accès aux images
        image_paths = []
        # Liste des étiquettes correspondantes ("Cat" ou "Dog")
        labels = []

        # Parcours des images de chats
        cat_image_files = glob.glob(cat_images_folder + "*.jpg")
        for cat_image_file in cat_image_files:
            image_paths.append(cat_image_file)
            labels.append("Cat")

        # Parcours des images de chiens
        dog_image_files = glob.glob(dog_images_folder + "*.jpg")
        for dog_image_file in dog_image_files:
            image_paths.append(dog_image_file)
            labels.append("Dog")
        # Liste des chemins d'accès aux images
    

        # Chargement des images et conversion en tableaux numpy
        images = []
        for path in image_paths:
            img = load_img(path, target_size=(224, 224))  # Ajustez la taille selon vos besoins
            img_array = img_to_array(img)
            images.append(img_array)
        images = np.array(images)
        predictions = model.predict(images)
        false_positives = []
        for i, prediction in enumerate(predictions):
            predicted_label = np.argmax(prediction)
            true_label = labels[i]
            if predicted_label == 0 and true_label == "Cat":
                false_positives.append(images[i])
            elif predicted_label == 1 and true_label == "Dog":
                false_positives.append(images[i])
        false_positives = np.array(false_positives)

        heatmap = np.mean(false_positives, axis=0)
        plt.figure(figsize=(10, 10))
        sns.heatmap(heatmap)
        plt.title('Heatmap des faux positifs')
        plt.axis('off')
        st.pyplot()"""
        
    with tab3:
        st.title("Connaitre les neuronnes ayant les plus gros poids")
        st.write("En analysant les poids, vous pouvez identifier les neurones importants qui ont des poids élevés ou significatifs. Les neurones avec des poids élevés ont un impact plus fort sur les prédictions du modèle et sont généralement responsables de la détection de caractéristiques spécifiques dans les images")
        
        st.image("poids_neuronnes.png")
        
        st.title("Pas lisible LOL ! Voici les noms des neuronnes les plus importants")
        # Charger le modèle pré-entraîné
        model = tf.keras.models.load_model('model_.h5')

        # Sélectionner la couche dont vous souhaitez obtenir les noms des neurones
        layer_index = 0  # Indice de la couche souhaitée
        layer = model.layers[layer_index]

        # Obtenir les noms des neurones dans la couche
        neuron_names = layer.get_config()['name']

        # Vérifier la taille des neurones dans la couche
        num_neurons = layer.get_weights()[0].shape[1]

        if num_neurons > 0:
            # Obtenir les poids de la couche
            weights = layer.get_weights()[0]

            # Calculer la somme des valeurs absolues des poids par neurone
            neuron_sums = tf.reduce_sum(tf.abs(weights), axis=0).numpy()

            # Obtenir les indices des neurones triés par importance décroissante
            sorted_indices = tf.argsort(neuron_sums, direction='DESCENDING').numpy()

            # Afficher les noms des neurones ayant les poids les plus élevés
            num_neurons_to_display = min(10, num_neurons)  # Nombre de neurones à afficher
            for i in range(num_neurons_to_display):
                neuron_index = sorted_indices[i]
                neuron_name = neuron_names + '_' + str(neuron_index)
                st.write(f"Neurone {neuron_name} a un poids élevé.")
        else:
            st.write("La couche sélectionnée n'a pas de neurones.")
    
        
    with tab5:
        st.title("Explication de Grad-CAM")

        st.header("À quoi ça sert ?")
        st.write("Grad-CAM (Gradient-weighted Class Activation Mapping) est une technique d'interprétabilité pour les réseaux de neurones convolutifs (CNN). Il permet de visualiser quelles parties d'une image sont importantes pour la prédiction faite par le réseau.")

        st.write("L'idée derrière Grad-CAM est de fournir une carte de chaleur (heatmap) qui indique les régions de l'image qui ont influencé le plus la décision du réseau. Par exemple, si vous avez un CNN entraîné pour reconnaître des chiens et des chats, Grad-CAM peut vous montrer les parties de l'image qui ont été déterminantes pour que le réseau dise qu'il y a un chien ou un chat.")
        
        st.header("Comment ça fonctionne ?")
        st.write("Voici les étapes de fonctionnement de Grad-CAM :")
        st.write("1. Charger une image et un modèle pré-entraîné.")
        st.write("2. Sélectionner la dernière couche convolutive avant la couche de classification comme couche cible.")
        st.write("3. Faire passer l'image à travers le modèle et enregistrer les gradients de la couche cible par rapport à la sortie.")
        st.write("4. Calculer les poids de gradient en moyennant les gradients spatiaux sur chaque canal.")
        st.write("5. Ponder les activations de la couche cible par les poids de gradient pour obtenir une carte d'activation.")
        st.write("6. Normaliser la carte d'activation et la superposer sur l'image originale pour visualiser les régions d'intérêt.")
        code_gradcam = "/home/matthieu/Images/Captures d’écran/Capture d’écran du 2023-05-22 10-05-49.png"
        st.image(code_gradcam, caption='GradCam', use_column_width=True)
        
        st.header("Limites de Grad-CAM")
        st.write("Bien que Grad-CAM soit un outil utile pour l'interprétabilité des CNN, il présente certaines limites :")
        st.write("- Grad-CAM ne tient pas compte des interactions entre les différentes parties de l'image.")
        st.write("- Il peut être sensible aux variations de l'échelle et de la rotation de l'image.")
        st.write("- Grad-CAM ne fournit pas d'explication causale, il montre seulement les régions d'intérêt.")