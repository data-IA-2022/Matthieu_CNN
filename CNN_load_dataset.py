import pandas as pd
import cv2
import pickle




path = "/home/matthieu/Formation_IA/Briefs/deep_learning_projet1/kagglecatsanddogs_5340/PetImages"


SIZE_list=[32, 64]

import os

for SIZE in SIZE_list:

    images = []
    labels = []
  
    for folder in os.listdir(path):
        for file_name in os.listdir(path+"/" + folder):

            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                try:
                    image = cv2.imread(path+"/" + folder + "/" + file_name)
                    images.append(cv2.resize(image, (SIZE, SIZE)))
                    labels.append(folder)
                                        
                except:
                    print(file_name, folder)
           
    data = {"images": images, "labels": labels}
    df = pd.DataFrame(data)
    
    
    with open("dataset_transfer" + str(SIZE)+".pickle", "wb") as f:
        pickle.dump(df, f)

    print("----------------------------------------")

