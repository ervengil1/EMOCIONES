import cv2
import glob
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ruta = r"C:\Users\Erven\Desktop\MCC\SEMESTRE 2 HECTORIN\EMOCIONES\\"
emotions = ["bored","engaged","excited","focused","interested"]
classifier_face = cv2.CascadeClassifier(r"C:\OPENCV\opencv\build\etc\lbpcascades\lbpcascade_frontalface.xml")

X = []
y = []

for index, emotion in enumerate(emotions):
    rutaEmocion = ruta+emotion
    for img in glob.glob(rutaEmocion+"\\*.jpeg"):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        faces = classifier_face.detectMultiScale(img)
        if len(faces) > 0:
            face = faces[0]
            img_face = img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            img_face = cv2.resize(img_face, (150, 150))
            cv2.imshow('frame', img_face)
            cv2.waitKey() 
            X.append(img_face)
            y.append(index)

pickle.dump(X, open("X.x","wb"))
pickle.dump(y, open("y.y","wb"))

# Cargar los datos de X.x y y.y
X = pickle.load(open("X.x", "rb"))
y = pickle.load(open("y.y", "rb"))

# Mostrar los primeros 5 ejemplos de imágenes junto con sus etiquetas
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[i], cmap='gray')  # Mostrar la imagen en escala de grises
    plt.title(emotions[y[i]])  # Obtener la emoción correspondiente según la etiqueta
    plt.axis('off')
plt.show()

X = np.array(X)
y = np.array(y)

# Crear un DataFrame de pandas con los datos
df = pd.DataFrame(X.reshape(X.shape[0], -1))  # Convertir cada imagen a una fila
df['Emotion'] = y  # Agregar la columna de emociones

# Renombrar las columnas para que sean más descriptivas (pixel1, pixel2, ..., pixel225)
column_names = ['pixel' + str(i) for i in range(1, X.shape[1] * X.shape[2] + 1)]
column_names.append('Emotion')
df.columns = column_names

# Guardar el DataFrame como un archivo CSV
df.to_csv('emotions_dataset.csv', index=False)
