import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ROCM_FUSION_ENABLE'] = '1'
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
import time


# Функция для создания модели
def create_cifar10_classifier(model_name, num_layers, num_filters, batch_size, epochs, dropout_rate):
    
    # Загрузка набора данных CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Нормализация значений пикселей изображений 
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Создание модели
    model = tf.keras.models.Sequential()
    
    # Добавление слоев свертки, подвыборки и регуляризации (Dropout)
    for _ in range(num_layers):
        model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(dropout_rate))

    # Преобразование данных и добавление полносвязного слоя
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout_rate))

    # Выходной слой с функцией активации Softmax для классификации на 10 классов
    model.add(layers.Dense(10, activation='softmax'))
    
    # Компиляция модели с оптимизатором Adam и функцией потерь sparse_categorical_crossentropy
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Запись времени начала обучения
    start_time = time.time()
    # Обучение модели на обучающем наборе данных
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    # Запись времени окончания обучения
    end_time = time.time()

    # Сохранение обученной модели на диск по указанному пути
    current_path = os.path.dirname(os.path.abspath(__file__))  
    model_path = (os.path.join(current_path, 'Models')) + '\\' + model_name + '.h5'
    
    model.save(model_path)
    
    # Текст с информацией о времени обучения
    text = "Время обучения: {:.2f} секунд".format(end_time - start_time)
    return text

    """
    # Оценка модели на тестовом наборе данных
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    """
# Функция для получение пути модели
def get_model_path(model_list):
    model_name = ''.join(model_list)
    current_path = os.path.dirname(os.path.abspath(__file__))  
    
    model_path = (os.path.join(current_path, 'Models')) + '\\' + model_name
   

    return model_path

# Функция для получение параметров модели
def get_model_parameters(model_list):

    # Загрузка тестовых данных cifar-10
    (_, _), (test_data, test_labels) = cifar10.load_data()

    # Препроцессинг данных
    test_data = test_data.astype('float32') / 255.0
    test_labels = to_categorical(test_labels)
    model_path = get_model_path(model_list)

    # Загрузка модели из файла
    model = tf.keras.models.load_model(model_path)

    # Компиляция модели
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Вычисление метрик на тестовом наборе данных
    loss, accuracy = model.evaluate(test_data, test_labels)
    result_loss_acc = f'Точность: {accuracy}, Потери: {loss}'
    return result_loss_acc



# Функция для определения изображений
def get_identify_image(model_path, image_path):
    class_names = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
    
    # Загрузка модели
    model = tf.keras.models.load_model(model_path)

    # Загрузка изображения
    image = Image.open(image_path)
    image = image.resize((32, 32))

    # Конвертирование изображения в массив numpy 
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Прогнозирование с помощью модели
    predictions = model.predict(image_array)

    # Получение прогнозируемого индекса класса
    predicted_class_index = np.argmax(predictions[0])

    # Получение названия класса
    class_name = class_names[predicted_class_index]

    # Получение точности прогноза
    prediction_accuracy = np.max(predictions[0]) * 100

    text = f'Предсказанный класс: {class_name}, Точность: {prediction_accuracy:.2f}%'
    return text