import streamlit as st
import tensorflow as tf
from PIL import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras_explication import keras_explication
from explication_modèle import explication_modèle
def home_page():
    st.title("Le but de cette appli :")
    st.write("Développer une application qui permet de détecter automatiquement des images d'animaux Chiens et Chats.")
    st.write("L'utilisateur doit pouvoir uploader une photo et l'application doit préciser de quel animal il s'agit ainsi que la probabilité de la classification.")
    st.write("Le classifieur sera développé avec Keras.")

    st.header("Bonus réalisés sur cette appli :")
    st.write("- Augmenter les usages de l'application par la classification d'autres animaux (reptiles) en utilisant un modèle pré-entrainé, grâce au Transfer Learning")
    st.write("- Tentative de compréhension du modèle avec GradCam")
    

    st.header("Livrables et Critères de performance")
    st.write("- Des scripts Python fonctionnels et commentés, déposés sur Github")
    st.write("- Une application Streamlit qui comporte au moins 3 onglets :")
    st.write("  1. Page d'accueil")
    st.write("  2. Page de chargement de l'image et de restitution des prédictions")
    st.write("  3. Page d'explication pédagogique des réseaux de neurones CNN")
    st.write("- Développement du CNN avec Keras")
    st.write("- Obtenir une accuracy >90% lors de la validation après apprentissage sur 25 epochs.")
    réussite_image = "/home/matthieu/Images/Captures d’écran/Capture d’écran du 2023-05-05 22-40-27.png"
    st.image(réussite_image, caption='Réussite image', use_column_width=True)
    st.write("- Présenter une application fonctionnelle et qui répond aux exigences.")
    st.write("- Etre en mesure d'expliquer le CNN à ma grand-mère.")
    mon_model = "/home/matthieu/Images/Captures d’écran/Capture d’écran du 2023-05-22 09-50-18.png"
    st.image(mon_model, caption='Réussite image', use_column_width=True)
    
    
    
def predict_image(image_path):
    # Charger votre modèle entraîné
    model = tf.keras.models.load_model('model_.h5')

    # Charger l'image à analyser
    img = image.load_img(image_path, target_size=(224, 224))

    # Convertir l'image en un tableau numpy
    x = image.img_to_array(img)

    # Ajouter une dimension supplémentaire pour représenter le lot de données (batch)
    x = np.expand_dims(x, axis=0)

    # Prétraiter l'image de la même manière que lors de l'entraînement de votre modèle
    x = preprocess_input(x)

    # Obtenir les prédictions pour l'image
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    # Afficher les probabilités pour chaque classe
    st.write("Prédictions : ", predictions)
    st.write("Prédictions : ", predicted_class)
    if predicted_class == 0:
        st.title("C'est un Chat")
    elif predicted_class == 1:
        st.title("C'est un Chien")


def prediction_page():
    st.title('Classification d\'images')
    
    # Affichage d'un message pour demander à l'utilisateur de télécharger une image
    st.write('Veuillez télécharger une image à classer')
    
    # Création d'un widget pour télécharger une image
    uploaded_file = st.file_uploader("Sélectionnez une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Enregistrer l'image localement
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Diviser l'écran en deux colonnes pour afficher les images côte à côte
        col1, col2 = st.columns(2)

        # Afficher l'image téléchargée dans la première colonne
        with col1:
            st.image("temp.jpg", caption='Image téléchargée', use_column_width=True)

        # Afficher la carte de chaleur dans la deuxième colonne
        with col2:
            prediction = predict_image("temp.jpg")
    
def cnn_page():
    st.title("Explication des Réseaux de Neurones Convolutifs (CNN)")

    st.header("Qu'est-ce qu'un CNN ?")
    st.write("Un réseau de neurones convolutifs (CNN) est un type de modèle d'apprentissage automatique, spécifiquement conçu pour traiter des données structurées en grille, comme des images. Il est largement utilisé pour des tâches telles que la classification d'images, la détection d'objets et la segmentation sémantique.")
    
    st.header("Quelques mots de vocabulaire :")
    st.write("-Un tenseur peut être considéré comme une matrice n-dimensionnelle. Dans le CNN ci-dessus, les tenseurs seront tridimensionnels à l'exception de la couche de sortie.")
    st.write("Un neurone peut être considéré comme une fonction qui prend plusieurs entrées et produit une seule sortie. Les sorties des neurones sont représentées ci-dessus sous la forme de cartes d'activation de couleur rouge → bleu.")
    st.write("Une couche est simplement une collection de neurones effectuant la même opération, avec les mêmes hyperparamètres.")
    st.write("Les poids et les biais du noyau, bien qu'ils soient propres à chaque neurone, sont ajustés pendant la phase d'apprentissage et permettent au classifieur de s'adapter au problème et à l'ensemble de données fournis. Ils sont représentés dans la visualisation avec une échelle de couleurs divergente jaune → verte. Les valeurs spécifiques peuvent être consultées dans la vue de formule interactive en cliquant sur un neurone ou en survolant le noyau/le biais dans la vue d'explication élastique de convolution.")
    st.write("Un CNN fournit une fonction de score différentiable, représentée sous forme de scores de classe dans la visualisation de la couche de sortie.")
    
    st.header("Comment fonctionne un CNN ?")
    st.write("Voici une explication simplifiée du fonctionnement d'un CNN :")
    st.write("1. Couches convolutives : Les couches convolutives sont responsables de l'extraction des caractéristiques visuelles de l'image. Elles utilisent des filtres (kernels) pour détecter des motifs locaux, tels que des bords, des textures ou des formes.")
    st.write("2. Couches de pooling : Les couches de pooling réduisent la dimensionnalité des caractéristiques extraites par les couches convolutives. Elles agrègent les informations en utilisant des opérations telles que le maximum ou la moyenne.")
    st.write("3. Couches entièrement connectées : Les couches entièrement connectées sont responsables de la classification finale. Elles prennent les caractéristiques extraites par les couches précédentes et les transforment en une sortie de probabilités pour chaque classe.")

    st.header("Pourquoi utiliser un CNN ?")
    st.write("Les CNN sont particulièrement adaptés pour l'analyse d'images en raison des raisons suivantes :")
    st.write("- Partage de poids : Les filtres convolutifs partagent leurs poids, ce qui permet au modèle de généraliser les caractéristiques apprises à différentes parties de l'image.")
    st.write("- Invariance aux translations : Les CNN sont capables de reconnaître des motifs indépendamment de leur position dans l'image grâce à l'utilisation de filtres convolutifs.")
    st.write("- Hiérarchie des caractéristiques : Les couches profondes d'un CNN apprennent des caractéristiques de plus en plus abstraites, ce qui permet de capturer des informations complexes et significatives.")

    st.header("Limites des CNN")
    st.write("Bien que les CNN soient largement utilisés et performants, ils présentent également certaines limites :")
    st.write("- Requiert des données annotées : Les CNN nécessitent souvent de grandes quantités de données annotées pour un entraînement efficace.")
    st.write("- Sensibles aux variations : Les CNN peuvent être sensibles aux variations telles que l'échelle, l'orientation ou l'éclairage des images.")
    st.write("- Interprétabilité limitée : Les modèles CNN sont souvent considérés comme des boîtes noires, ce qui rend difficile l'interprétation des décisions prises par le modèle.")

def explication_tf():
    st.title("Explication du Transfer Learning")

    st.header("Fonctionnement")
    st.write("Le Transfer Learning est une technique utilisée en apprentissage automatique (machine learning) qui permet de transférer les connaissances apprises par un modèle pré-entraîné sur une tâche spécifique à une nouvelle tâche similaire. Voici comment cela fonctionne :")
    
    explication_transfer = "/home/matthieu/explication_tranfer.jpg"
    st.image(explication_transfer, caption='Transfer', use_column_width=True)
    
    st.write("1. Pré-entraînement initial : Un modèle est entraîné sur un ensemble de données volumineux et représentatif, souvent sur une tâche de classification générale, tel que le jeu de données ImageNet.")
    st.write("2. Extraction de caractéristiques : Les couches convolutives du modèle pré-entraîné capturent des caractéristiques visuelles génériques et abstraites.")
    st.write("3. Congélation des poids : Les poids des couches convolutives sont généralement figés pour empêcher leur modification lors de l'entraînement de la nouvelle tâche.")
    st.write("4. Ajout de nouvelles couches : Une nouvelle tête de classification est ajoutée au modèle pour correspondre à la nouvelle tâche. Ces couches sont initialisées aléatoirement.")
    st.write("5. Entraînement du modèle : Seules les nouvelles couches sont entraînées sur un ensemble de données spécifique à la nouvelle tâche.")
    st.write("6. Fine-tuning (réglage fin) optionnel : Si l'ensemble de données spécifique à la nouvelle tâche est suffisamment grand, les poids des couches convolutives peuvent être ajustés (fine-tuning) pour améliorer les performances du modèle.")

    mon_code = "/home/matthieu/Images/Captures d’écran/Capture d’écran du 2023-05-22 10-59-55.png"
    st.image(mon_code, caption='Mon_code', use_column_width=True)


    st.header("Avantages et utilité")
    st.write("Le Transfer Learning présente plusieurs avantages et utilités :")
    st.write("- Réduction du temps d'entraînement : Étant donné que les couches convolutives sont pré-entraînées, le modèle peut être entraîné plus rapidement sur la nouvelle tâche.")
    st.write("- Besoin de moins de données : Le modèle peut généraliser à de nouvelles tâches même avec moins de données d'entraînement, car il a déjà appris des caractéristiques visuelles générales.")
    st.write("- Amélioration des performances : Le Transfer Learning permet d'obtenir de meilleures performances sur la nouvelle tâche, surtout lorsque les données d'entraînement sont limitées.")
    st.write("- Adaptation à de nouveaux domaines : Le modèle pré-entraîné peut être utilisé comme point de départ pour différentes tâches et domaines d'application, en bénéficiant des connaissances acquises sur des données préalablement traitées.")

    st.header("Limites")
    st.write("Malgré ses avantages, le Transfer Learning présente également certaines limites :")
    st.write("- Spécificité de la tâche : Le modèle pré-entraîné peut ne pas être adapté exactement à la nouvelle tâche, ce qui peut limiter ses performances.")
    st.write("- Disponibilité des données : Si les données de la nouvelle tâche diffèrent considérablement de celles sur lesquelles le modèle a été pré-entraîné, le Transfer Learning peut être moins efficace.")
    st.write("- Transfert inapproprié : Si la nouvelle tâche est très différente de la tâche d'origine, le transfert de connaissances peut ne pas être pertinent.")
    st.write("- Risque de biais : Si le modèle pré-entraîné est biaisé envers certaines caractéristiques ou catégories, cela peut influencer les performances sur la nouvelle tâche.")

def predict_image_transfer(image_path):
    # Charger votre modèle entraîné
    model = tf.keras.models.load_model('modele_transfer_learning.h5')
    
    # Charger l'image à analyser
    img = image.load_img(image_path, target_size=(64, 64))

    # Convertir l'image en un tableau numpy
    x = image.img_to_array(img)

    # Ajouter une dimension supplémentaire pour représenter le lot de données (batch)
    x = np.expand_dims(x, axis=0)

    # Prétraiter l'image de la même manière que lors de l'entraînement de votre modèle
    x = preprocess_input(x)

    # Obtenir les prédictions pour l'image
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    # Afficher les probabilités pour chaque classe
    st.write("Prédictions : ", predictions)
    st.write("Prédictions : ", predicted_class)
    if predicted_class == 0:
        st.write("C'est un chat")
    elif predicted_class == 1:
        st.write("(C'est peut-être un caméléon)")
    elif predicted_class == 2 :
        st.write("(C'est peut-être un crocodile/alligator)")
    elif predicted_class == 3:
        st.write("(C'est peut-être un chien)")
    elif predicted_class == 4 :
        st.write("(C'est peut-être un grenouille)")
    elif predicted_class == 5 :
        st.write("(C'est peut-être un gecko)")
    elif predicted_class == 6 :
        st.write("(C'est peut-être un iguane)")
    elif predicted_class == 7 :
        st.write("(C'est peut-être un lézard)")
    elif predicted_class == 8 :
        st.write("(C'est peut-être un salamandre)")
    elif predicted_class == 9 :
        st.write("(C'est peut-être un serpent)")
    elif predicted_class == 10 :
        st.write("(C'est peut-être un crapaud)")
    elif predicted_class == 11 :
        st.write("(C'est peut-être un tortue)")


def transfer_mode():
    st.title('Classification d\'images')
    
    # Affichage d'un message pour demander à l'utilisateur de télécharger une image
    st.write('Veuillez télécharger une image à classer')

    # Charger l'image à partir de l'utilisateur
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
    # Enregistrer l'image localement
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Diviser l'écran en deux colonnes pour afficher les images côte à côte
        col1, col2 = st.columns(2)
        
        # Afficher l'image téléchargée dans la première colonne
        with col1:
            st.image("temp.jpg", caption='Image téléchargée', use_column_width=True)

        # Afficher la carte de chaleur dans la deuxième colonne
        with col2:
            prediction = predict_image_transfer("temp.jpg")
            predicted_class = np.argmax(prediction)
            

def predict_image2(image_path):
    # Charger votre modèle entraîné
    model = tf.keras.models.load_model('model_.h5')

    # Charger l'image à analyser
    img = image.load_img(image_path, target_size=(224, 224))

    # Convertir l'image en un tableau numpy
    x = image.img_to_array(img)

    # Ajouter une dimension supplémentaire pour représenter le lot de données (batch)
    x = np.expand_dims(x, axis=0)

    # Prétraiter l'image de la même manière que lors de l'entraînement de votre modèle
    x = preprocess_input(x)

    # Obtenir les prédictions pour l'image
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    # Initialiser GradCAM avec le modèle et la couche de sortie souhaitée
    explainer = GradCAM()

    # Obtenir le gradient des activations pour la classe prédite
    grid = explainer.explain((x, None), model, class_index=np.argmax(predictions))

    # Afficher la heatmap avec les régions les plus importantes pour la prédiction
    st.write("Carte de chaleur : ")
    st.image(grid, use_column_width=True)
    st.write("Prédictions : ", predictions)
    st.write("Prédictions : ", predicted_class)
    if predicted_class == 0:
        st.title("C'est un Chat")
    elif predicted_class == 1:
        st.title("C'est un Chien")

def comprehension_page():
    st.title('Classification d\'images')
    
    # Affichage d'un message pour demander à l'utilisateur de télécharger une image
    st.write('Veuillez télécharger une image à classer')
    
    st.title("Analyse d'image avec GradCAM")
    model = tf.keras.models.load_model('model_.h5')
    # Charger l'image à partir de l'utilisateur
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Enregistrer l'image localement
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Diviser l'écran en deux colonnes pour afficher les images côte à côte
        col1, col2 = st.columns(2)
        
        # Afficher l'image téléchargée dans la première colonne
        with col1:
            st.image("temp.jpg", caption='Image téléchargée', use_column_width=True)

        # Afficher la carte de chaleur dans la deuxième colonne
        with col2:
            prediction = predict_image2("temp.jpg")
            
        
        



# Création de l'application Streamlit
def main():
    st.set_page_config(page_title='Classification d\'images', page_icon=':shark:', layout='wide')
    
    # Ajout d'une barre latérale pour la navigation entre les onglets
    st.sidebar.title('Navigation')
    pages = ['Accueil','Explication Keras', 'Classification d\'images', 'Réseaux de neurones CNN','Explication Transfer Learning','Modèle de Tranfer learning',"Comprèhension du modèle", "GradCam"]
    page = st.sidebar.radio('Choisissez une page', pages)
    
    # Affichage de la page sélectionnée
    if page == 'Accueil':
        home_page()
    elif page == 'Explication Keras':
        keras_explication()
    elif page == 'Classification d\'images':
        prediction_page()
    elif page == 'Réseaux de neurones CNN':
        cnn_page()
    elif page == 'Explication Transfer Learning':
        explication_tf()
    elif page == 'Modèle de Tranfer learning':
        transfer_mode()
    elif page == 'Comprèhension du modèle':
        explication_modèle()
    elif page == 'GradCam':
        comprehension_page()
    
if __name__ == '__main__':
    main()