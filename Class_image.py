import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from PIL import Image
import numpy as np

# Загрузка обученной модели
model = tf.keras.models.load_model(r'C:\Users\osevi\OneDrive\Documents\doplom\Models\geeksforgeeks.h5')

# Загрузка своего изображения
image_path = r'C:\Users\osevi\OneDrive\Documents\doplom\Test_image\horse1.webp'
image = Image.open(image_path)
image = image.resize((32, 32))  # Изменяем размер изображения под модель

# Преобразование изображения в массив numpy и нормализация
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)  # Добавляем размерность для батча

# Предсказание на модели
predictions = model.predict(image_array)

# Получение индекса предсказанного класса
predicted_class_index = np.argmax(predictions[0])

# Вывод результата
print(f'Предсказанный класс номер: {predicted_class_index}')