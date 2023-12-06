import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import torch
import requests
import os


st.set_page_config(page_title="TechScan NN", layout="wide")
# st.title("TechScan NN \n Reconocimiento de Texto en Imágenes de Equipos Eléctricos")
st.title("TechScan NN")
st.title("Reconocimiento de Texto en Imágenes de Equipos Eléctricos")
st.write("Esta aplicación permite clasificar imágenes en 4 clases diferentes (trasformadores, placas de transformadores, características de postros y placas de postes). Adicionalmente, si la imagen pertence a las clases de placas, se ejecutan modelos de detección de objetos y aplicación de máscaras para reconocer el texto. Se realizar el procesamiento una imagen a la vez.")


# carga modelo de clasificación streamlit
if not os.path.exists("models/clf_model.h5"):
    model_clsf_path = 'https://github.com/j2sanabriam/Proyecto_DL/blob/16ef8c600adf08b3ed55e2a50b2cb5a7485f1a2a/App/models/clf_model.h5?raw=true'
    response = requests.get(model_clsf_path)
    with open("models/clf_model.h5", "wb") as file:
        file.write(response.content)


if 'file' not in st.session_state:
    st.session_state['file'] = None

if not st.session_state['file']:

    # Opción para cargar imagen
    uploaded_photo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    # Acciones al ejecutar botón Realizar Predicción
    if st.button("Realizar Reconocimiento") and uploaded_photo is not None:

        st.subheader("Imagen Original")
        photo = Image.open(uploaded_photo)
        st.image(photo, use_column_width=True)
        st.session_state['file'] = photo

    # MODELO CLASIFICACIÓN
        # Ejecución página localmente
        # model_clsf_path = './models/clf_model.h5'
        # model_clsf = tf.keras.models.load_model(model_clsf_path)

        # Ejecución página streamlit
        model_clsf = tf.keras.models.load_model("models/clf_model.h5")

        # Cargar y preparar la imagen
        img = image.load_img(uploaded_photo, target_size=(128, 128), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array.reshape(1, -1)  # Aplanar la imagen a un vector
        img_array /= 255.0

        # Hacer la predicción
        prediction = model_clsf.predict(img_array)
        # Procesar la predicción según sea necesario
        predicted_class = np.argmax(prediction, axis=1)
        # Imprime o devuelve la clase predicha
        st.subheader("Predicción de Clase: " + str(predicted_class[0]))

    # VALIDA TIPOS DE FOTOS
        if predicted_class == 0:
            st.write("Placa de Poste")

        # APLICACIÓN MODELO YOLO SOBRE PLACAS DE POSTES
            # YOLO_model_path = "./models/YOLO_model.pt"
            YOLO_model_path = "models/YOLO_model.pt"
            # Cargar el modelo
            YOLO_model = torch.hub.load("ultralytics/yolov5:master", "custom", path=YOLO_model_path)

            # Realizar la detección en la imagen
            st.subheader("Imagen Aplicando YOLO v5")
            results = YOLO_model(photo)
            image_with_boxes = results.render()[0]
            st.image(image_with_boxes, use_column_width=True)

        # APLICACIÓN MODELO MÁSCARA PARA POSTES
            # modelo_ruta_mask_poste = './models/postes_320_320.h5'
            modelo_ruta_mask_poste = 'models/postes_320_320.h5'
            # Cargar el modelo
            postes_mask_model = load_model(modelo_ruta_mask_poste)

            # Predicción Máscaras Postes
            nueva_imagen = photo.resize((320, 320))
            # Convertir la imagen a un arreglo numpy y normalizar los valores de píxeles
            nueva_imagen_array = np.array(nueva_imagen)
            nueva_imagen_array = np.expand_dims(nueva_imagen_array, axis=0)

            mask_poste_pred = postes_mask_model.predict(nueva_imagen_array)

            # Supongamos que predicciones[0] contiene la máscara de segmentación
            mascara_predicha = mask_poste_pred[0]
            # Seleccionar el índice del canal con la probabilidad máxima para cada píxel
            clase_predicha = np.argmax(mascara_predicha, axis=-1)
            # Identificar la clase que se pinta de amarillo (supongamos que es la clase 1)
            clase_eliminar = 1
            # Crear una máscara booleana para la clase que se quiere eliminar
            mascara_eliminar = (clase_predicha == clase_eliminar)

            # Crear una máscara booleana para las otras dos clases
            mascara_otras_clases = ~mascara_eliminar
            # Obtener la imagen de las otras dos clases
            imagen_otras_clases = nueva_imagen_array[0].copy()
            imagen_otras_clases[mascara_eliminar, :] = 0  # Establecer a cero la clase que se quiere eliminar

            # Encontrar contornos en la máscara de las otras dos clases
            contornos, _ = cv2.findContours(mascara_otras_clases.astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            # Dibujar rectángulos alrededor de los contornos
            imagen_con_bounding_box = imagen_otras_clases.copy()
            for contorno in contornos:
                x, y, w, h = cv2.boundingRect(contorno)
                cv2.rectangle(imagen_con_bounding_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Visualizar la imagen con bounding box
            fig, ax = plt.subplots()
            imagen_plot = ax.imshow(imagen_con_bounding_box, cmap='viridis')

            # Mostrar el gráfico en Streamlit
            st.subheader("Imagen de Poste con Máscara")
            st.pyplot(fig)

        elif predicted_class == 1:
            st.write("Placa de Trasformador")

        # APLICACIÓN MODELO MÁSCARA PARA POSTES
            # modelo_ruta_mask_placa = './models/placas_segmentation_resize_320_320_resize.h5'
            modelo_ruta_mask_placa = 'models/placas_segmentation_resize_320_320_resize.h5'
            # Cargar el modelo
            placa_mask_model = load_model(modelo_ruta_mask_placa)

            # Cargar la imagen y redimensionarla a 320x320
            nueva_imagen2 = photo.resize((320, 320))
            # Convertir la imagen a un arreglo numpy y normalizar los valores de píxeles
            nueva_imagen_array2 = np.array(nueva_imagen2)
            nueva_imagen_array2 = np.expand_dims(nueva_imagen_array2, axis=0)

            predicciones2 = placa_mask_model.predict(nueva_imagen_array2)

            # Visualizar la imagen original con la máscara predicha superpuesta
            fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
            # Mostrar la imagen original
            imagen_plot2 = ax2.imshow(nueva_imagen_array2[0])

            # Superponer la máscara predicha (tercera imagen) en la imagen original
            ax2.imshow(predicciones2[0], alpha=0.5,
                       cmap='gray')  # Ajusta el valor de alpha según sea necesario para la transparencia
            # Configurar el título y desactivar los ejes
            ax2.set_axis_off()

            # Mostrar el gráfico en Streamlit
            st.subheader("Imagen de Placa Transformador con Máscara")
            st.pyplot(fig2)

        elif predicted_class == 2:
            st.write("Poste (características)")
        elif predicted_class == 3:
            st.write("Trasformador")

        st.session_state['file'] = None

