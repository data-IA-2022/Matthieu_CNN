from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

SIZE = 64

with open("dataset_transfer" + str(SIZE) + ".pickle", "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame.from_dict(data)

train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['labels'])

folder_names = ['Cat', 'Chameleon', 'Crocodile_Alligator', 'Dog', 'Frog', 'Gecko', 'Iguana', 'Lizard', 'Salamander', 'Snake', 'Toad', 'Turtle_Tortoise']
class_labels = {i: folder_names[i] for i in range(len(folder_names))}
train_df['labels'] = train_df['labels'].map(class_labels)
test_df['labels'] = test_df['labels'].map(class_labels)

print(folder_names)
print(class_labels)
print(train_df)
print(test_df)
