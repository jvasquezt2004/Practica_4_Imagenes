import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


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
    # Suavizar las imágenes
    a_image_filtered = a_image.filter(ImageFilter.GaussianBlur(1))
    m_image_filtered = m_image.filter(ImageFilter.GaussianBlur(1))

    # Calcular la diferencia absoluta entre las imágenes suavizadas
    difference_image = ImageChops.difference(a_image_filtered, m_image_filtered)
    # Convertir la diferencia en escala de grises
    difference_image = difference_image.convert("L")

    # Aplicar el umbral para crear la máscara binaria
    binary_mask = difference_image.point(lambda p: 255 if p > threshold else 0)

    # Aplicar operaciones morfológicas para mejorar la máscara
    binary_mask = binary_mask.filter(ImageFilter.MinFilter(3))  # Erosión
    binary_mask = binary_mask.filter(ImageFilter.MaxFilter(5))  # Dilatación

    # Podemos invertir la máscara si es necesario
    # binary_mask = ImageOps.invert(binary_mask)

    return binary_mask


def combine_images(a_image, f_image, mask_u):
    # Asegurar que todas las imágenes tienen el mismo tamaño
    if not (a_image.size == f_image.size == mask_u.size):
        print("Error: Todas las imágenes deben tener el mismo tamaño.")
        return None

    # NOT U
    mask_u_inverted = ImageOps.invert(mask_u)

    # F AND NOT U
    background_part = ImageChops.multiply(
        f_image.convert("RGBA"), mask_u_inverted.convert("RGBA")
    )

    # A AND U
    person_part = ImageChops.multiply(a_image.convert("RGBA"), mask_u.convert("RGBA"))

    # R := (F AND NOT U) OR (A AND U)
    combined_image = ImageChops.add(background_part, person_part)
    return combined_image


# Cargar imágenes de fondo y la imagen A
a_image = Image.open(
    "img/Mask4.jpeg"
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

        # Cargar la nueva imagen de fondo F
        f_image = Image.open(
            "img/NewBackground.jpg"
        )  # Reemplaza con la ruta a la nueva imagen de fondo

        # Asegúrate de que la imagen F y la máscara U tengan el mismo tamaño
        if f_image.size != mask_u.size:
            print(
                "Error: La nueva imagen de fondo y la máscara U no son del mismo tamaño."
            )
        else:
            # Combinar imágenes
            r_image = combine_images(a_image, f_image, mask_u)
            if r_image is not None:
                # Mostrar la imagen combinada
                r_image.show()
                # Opcional: guardar la imagen combinada
                # r_image.save('img/CombinedImage.jpeg')  # Reemplaza con la ruta donde quieres guardar la imagen combinada
