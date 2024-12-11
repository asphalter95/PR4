# Распознавание рукописных символов EMNIST

## 1. Описание решения
_Опишите в этом разделе задание и ваше решение:_
- В данном проекте решается задача классификации
- На вход подаёся массив размерностью 28*28, а возвращается label предсказания
- Модель SVC с стандартными гиперпараметрами показала очень хорошую и стабильную метрику на тренировочной и тестовой выборках
- Метрика accuracy на тестовых данных = 0.84+


## 2. Установка и запуск сервиса

_Опишите в этом разделе, как запустить ваше решение, где должен запуститься сервис, как им пользоваться. Если вы хотите сообщить пользователям и проверяющим дополнительную информацию, сделайте это здесь._
Чтобы получить изображение и вставить его значением для отправки в поле "image", нужен код
```python
import gzip
with gzip.open('gzip/emnist-balanced-test-images-idx3-ubyte.gz', 'rb') as f:
    data = f.read()
magic_number = int.from_bytes(data[0:4], byteorder='big')
num_images = int.from_bytes(data[4:8], byteorder='big')
num_rows = int.from_bytes(data[8:12], byteorder='big')
num_cols = int.from_bytes(data[12:16], byteorder='big')
images = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, num_rows, num_cols)
image = images[0]
import re
numbers = re.sub(r'[^\d]', ' ', str(image)).split()
```
```bash
git clone https://github.com/asphalter95/PR4.git
cd ./myapp
docker build -t myapp .
docker run -p 8000:8000 myapp
```
После запуска сервис будет доступен по адресу http://localhost:8000

В postman нужно прописать localhost:8000/predict, телом прописать изображение в виде
{"image":numbers}