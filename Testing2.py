import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# prepare data
input_dir = 'clf-data'
# categories = ['Albrecht_Duhrer', 'Alfred_Sisley', 'Amedeo_Modigliani','Andrei_Rublev','Andy_Warhol']
categories = ['Albrecht_Duhrer','Alfred_Sisley','Amedeo_Modigliani','Andrei_Rublev','Andy_Warhol','Camille_Pissarro','Caravaggio','Claude_Monet','Diego_Rivera','Diego_Velazquez','Edgar_Degas','Edouard_Manet','El_Greco','Eugene_Delacroix','Francisco_Goya','Frida_Kahlo','Georges_Seurat','Giotto_di_Bondone','Gustav_Klimt','Gustave_Courbet','Henri_de_Toulouse-Lautrec','Henri_Matisse','Hieronymus_Bosch','Jackson_Pollock','Jan_van_Eyck','Joan_Miro','Kazimir_Malevich','Leonardo_da_Vinci','Marc_Chagall','Michelangelo','Mikhail_Vrubel','Pablo_Picasso','Paul_Cezanne','Paul_Gauguin','Paul_Klee','Peter_Paul_Rubens','Pierre-Auguste_Renoir','Piet_Mondrian','Pieter_Bruegel','Raphael','Rembrandt','Rene_Magritte','Salvador_Dali','Sandro_Botticelli','Titian','Vasiliy_Kandinskiy','Vincent_van_Gogh','William_Turner']

data = []
labels = []

shapesizeNih = 512

target_size = (shapesizeNih, shapesizeNih, 1)
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        print("Processing", img_path)
        try:
            img = imread(img_path)
            
            # Convert to grayscale if the image has multiple channels
            if len(img.shape) == 3:
                img = np.mean(img, axis=-1, keepdims=True).astype(np.uint8)
            
            # Convert to a specific bit depth or data type
            img = img.astype(np.uint8)  # Assuming 8-bit depth
            
            img = resize(img, target_size)
            
            # Print the dimensions after resizing
            print("Dimensions after resizing:", img.shape)
            
            # Print the shape before appending
            print("Shape before appending:", img.shape)
            
            print("")
            data.append(img)
            labels.append(category)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Rest of your code

data = np.asarray(data)
labels = np.asarray(labels)

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Convert labels to one-hot encoding
labels_one_hot = to_categorical(labels_encoded)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.2, shuffle=True, stratify=labels_one_hot)

# Build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(shapesizeNih, shapesizeNih, 1)))  # Assuming input images are grayscale
model.add(Dense(128, activation='relu'))
model.add(Dense(48, activation='softmax'))  # Assuming 48 categories

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_test, y_test)
print(f'Accuracy on test set: {score[1] * 100:.2f}%')

# Save the model
model.save('./keras_model.h5')
