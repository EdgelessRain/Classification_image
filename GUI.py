import PySimpleGUI as sg
import os
import CNN 
from PIL import Image
import io

model_path = '' #путь модели
image_path = '' #путь изображения

#функция для изменения размера изображения для вывода
def resize_image(img, target_size):
    img.thumbnail(target_size, Image.NONE)  
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

#функция вывода главного меню
def make_window_menu():
    layout = [
            [sg.Button('Определить изображение', key = '-BUT1-', button_color = ('black','yellow'),  size= (12, 3)), 
             sg.Button('Создать модель', key = '-BUT2-', button_color = ('black','yellow'),  size= (12, 3)),
             sg.Button('Выбрать модель', key = '-BUT3-', button_color = ('black','yellow'),  size= (12, 3))
             ],
            [sg.Text()],
            [sg.Button('Выход', button_color= ('black','red'), size= (12, 3) )]]
    return sg.Window('Главное меню', layout, finalize=True, size= (450, 500),  element_justification='c')

#функция вывода меню для определения изображения
def make_window_identify_image():
    target_size = (300, 300)  # Целевой размер для изображения
    layout = [
        [sg.Text('Выберите изображение для загрузки')],
        [sg.Input(key='-FILE-'), sg.FileBrowse(file_types=(("Image Files", "*.jpg;*.png;*.webp"),))],
        [sg.Button('Показать изображение'), sg.Button('Определить класс изображения'), sg.Button('Отмена')],
        [sg.Image(key='-IMAGE-', size=target_size)],
        [sg.Text(size=(40, 1), key='-CLASS-', auto_size_text=True)],
        [sg.Button('Назад')]]
    return sg.Window('Определение изображений', layout, finalize=True, size= (450, 500), element_justification='c')

#функция вывода меню для выбора модели
def make_window_сhoose_model():
    current_path = os.path.dirname(os.path.abspath(__file__))  #  путь к текущей директории
    folder_path = os.path.join(current_path, 'Models')  # путь к папке с моделями 
    file_list = os.listdir(folder_path)  # список всех моделей в папке
    layout = [
            [sg.Button('Выбрать модель', key='model', button_color=('black', 'yellow'), size=(12, 3)), 
             sg.Listbox(values=file_list, size=(50, 10), key='-FILE LIST-')],
            [sg.Button('Параметры текущей модели', key='get_par', button_color=('black', 'yellow'), size=(12, 3)),
             sg.Text(key='parameters')],
            [sg.Text()],
            [sg.Button('Назад')]
    ]
    return sg.Window('Выбор модели', layout, finalize=True, size=(450, 500), element_justification='c')

#функция вывода меню для создания модели
def make_window_create_model():
    layout = [
        [sg.Text('Введите название модели:'), sg.InputText(key='model_name')],
        [sg.Text('Выбор параметров нейросети')],
        [sg.Text('Максимальное количество фильтров:'), sg.InputCombo(['32', '64', '128', '256'], key='filters')],
        [sg.Text('Количество примеров обучения:'), sg.InputCombo(['32', '64', '128', '256'], key='batch_size')],
        [sg.Text('Процент отсева:'), sg.InputCombo([str(round(i * 0.05, 2)) for i in range(0, 21)], key='dropout')],
        [sg.Text('Количество циклов:'), sg.InputCombo([str(i) for i in range(1, 51)], key='epochs')],
        [sg.Text('Количество слоев:'), sg.InputCombo([str(i) for i in range(1, 4)], key='layers')],
        [sg.Button('Создать')],
        [sg.Text('', key='time')],
        [sg.Button('Назад')]]
    return sg.Window('Обучение нейросети', layout, finalize=True, size= (450, 500),  element_justification='c')
 
#функция вывода окон приложения на экран
def make_window():
    global model_path
    global image_path
    sg.theme('DarkAmber')
    window_menu, window_identify_image, window_сhoose_model, window_create_model = make_window_menu(), None, None, None

    while True:
        window, event, values = sg.read_all_windows()

        if window == window_menu and event in (sg.WIN_CLOSED, 'Выход'):
            break

        elif window == window_menu:
            if event == '-BUT1-' and model_path == '':
                sg.popup_error_with_traceback(f'не выбрана модель!')

            elif event == '-BUT1-':
                window_menu.hide()
                window_identify_image = make_window_identify_image()    

            elif event == '-BUT2-':
                window_menu.hide()
                window_create_model = make_window_create_model() 

            elif event == '-BUT3-':
                window_menu.hide()
                window_сhoose_model = make_window_сhoose_model()

        elif window == window_identify_image:
            filename = values['-FILE-']
            image_path = values['-FILE-']
            if (event == 'Определить класс изображения' and image_path == '') or (event == 'Показать изображение' and image_path == ''):
                sg.popup_error_with_traceback(f'не выбрано изображение!')


            elif event == 'Показать изображение':

                if filename:
                    img = Image.open(filename)
                    resized_img = resize_image(img, (300, 300))  # Изменяем размер изображения
                    window['-IMAGE-'].update(data=resized_img)



            elif event == 'Определить класс изображения':
                print(image_path)
                window['-CLASS-'].update(CNN.get_identify_image(model_path, image_path))

            elif event in (sg.WIN_CLOSED, 'Назад'):
                window_identify_image.close()
                window_menu.un_hide()    

        elif window == window_сhoose_model:
            if event == 'model' and values['-FILE LIST-'] == []:
                sg.popup_error_with_traceback(f'не выбрана модель!')
            elif event == 'model':
                print(values['-FILE LIST-'])
                model_name = ''.join(values['-FILE LIST-'])
                current_path = os.path.dirname(os.path.abspath(__file__))  # определяем путь к текущей директории, где находится файл с данным кодом
                models_path = os.path.join(current_path, 'Models')
                model_path = models_path + '\\' + model_name 
                window_сhoose_model.hide()
                window_menu = make_window_menu()

            elif event == 'get_par':
                if values['-FILE LIST-'] == []:
                    window_сhoose_model['parameters'].update('Вы не выбрали модель!') 
                else:
                    window_сhoose_model['parameters'].update('Подождите') 
                    window_сhoose_model['parameters'].update(CNN.get_model_parameters(values['-FILE LIST-']))    

            elif event in (sg.WIN_CLOSED, 'Назад'):
                window_сhoose_model.close()
                window_menu.un_hide()             

        elif window == window_create_model:
            if event == sg.WINDOW_CLOSED:
                break

            if event == 'Создать':
                window['Создать'].update(disabled=True)
                if values['model_name'] == "" or values['layers'] == "" or values['filters'] == "" or values['batch_size'] == "" or values['epochs'] == "" or values['dropout'] == "":
                    sg.popup_error_with_traceback(f'не выбраны все параметры!')
                else:
                    window['Создать'].update('Модель создается, подождите...')
                    window.refresh()
                    window['time'].update(CNN.create_cifar10_classifier(values['model_name'], int(values['layers']), int(values['filters']),  int(values['batch_size']), int(values['epochs']), float(values['dropout']),))
                    window['Создать'].update('Модель готова')
            elif event in (sg.WIN_CLOSED, 'Назад'):
                window_create_model.close()
                window_menu.un_hide()  
                
    window.close()