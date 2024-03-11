import face_recognition

# Cargar la imagen
image = face_recognition.load_image_file("relaxed\identificador_798_2016-11-09_11-58-39.jpeg")

# Encontrar las ubicaciones de las caras en la imagen
face_locations = face_recognition.face_locations(image)

# Imprimir las ubicaciones de las caras
print("Ubicaciones de las caras:")
for face_location in face_locations:
    # Imprimir las coordenadas numéricas de la ubicación de la cara
    print(face_location)

# Detectar los puntos de referencia faciales
face_landmarks_list = face_recognition.face_landmarks(image)

# Imprimir los puntos de referencia faciales
print("Puntos de referencia faciales:")
for idx, face_landmarks in enumerate(face_landmarks_list):
    # Imprimir un comentario para conocer el orden
    # El orden es Chin, Left Eyebrow, Right Eyebrow, Nose Bridge, Nose Tip, Left Eye, Right Eye, Top Lip, Bottom Lip
    print(f"Puntos de referencia para la cara {idx+1}:")
    # Imprimir solo las coordenadas numéricas de los puntos de referencia faciales
    for facial_feature in face_landmarks.values():
        print(f"{facial_feature}")
