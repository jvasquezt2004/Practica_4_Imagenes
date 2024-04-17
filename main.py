import numpy as np
from PIL import Image, ImageChops


def calculate_model(images):
    image_quantity = len(images)
    # Convertir imágenes a arrays y calcular la suma
    sum_image = np.sum(np.stack([np.array(image) for image in images]), axis=0)
    # Calcular la imagen media
    mean_image = sum_image / image_quantity
    # Calcular la varianza y normalizar antes de convertirla a imagen
    variance = (
        np.sum([np.square(np.array(image) - mean_image) for image in images], axis=0)
        / image_quantity
    )
    # Normalizar la varianza para que esté en el rango de los valores de uint8
    variance_image = 255 * (variance / np.max(variance))
    return Image.fromarray(np.uint8(mean_image)), Image.fromarray(
        np.uint8(variance_image)
    )


def calculate_mask(a_image, m_image, threshold=30):
    # Calcular la diferencia absoluta entre las imágenes
    difference_image = ImageChops.difference(a_image, m_image)
    # Convertir la diferencia en escala de grises
    difference_image = difference_image.convert("L")
    # Aplicar el umbral para crear la máscara binaria
    binary_mask = difference_image.point(lambda p: 255 if p > threshold else 0)
    return binary_mask


# Cargar imágenes
a_image = Image.open(
    "img/Mask.jpeg"
)  # Reemplaza con la ruta a tu imagen donde apareces
image_paths = [
    "img/Background1.jpeg",
    "img/Background2.jpeg",
    "img/Background3.jpeg",
]  # Reemplaza con las rutas a tus imágenes de fondo
background_images = [Image.open(path) for path in image_paths]

# Asegurarse de que las imágenes se cargaron correctamente
if any(img is None for img in background_images):
    print(
        "Error cargando alguna de las imágenes de fondo. Por favor verifica las rutas."
    )
else:
    # Calcular el modelo de imagen (media y varianza)
    mean_image, variance_image = calculate_model(background_images)
    # Asegurar que la imagen A y la media tienen el mismo tamaño
    if a_image.size != mean_image.size:
        print("Error: las imágenes no son del mismo tamaño.")
    else:
        # Calcular la máscara U
        mask_u = calculate_mask(
            a_image, mean_image, threshold=30
        )  # Ajusta el umbral según sea necesario
        # Mostrar la máscara
        mask_u.show()
        # Opcional: guardar la máscara
        # mask_u.save('path_to_save_mask.jpg')  # Reemplaza con la ruta donde quieres guardar la máscara
