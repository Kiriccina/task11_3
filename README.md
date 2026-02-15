# task11_4
#В этом задании вы будете работать с датасетом, который содержит данные о характеристиках автомобилей и их стоимости. Ваша задача - построить модель линейной регрессии для предсказания цен автомобилей (столбец MSRP), а затем сравнить её точность с так называемой "глупой моделью"#

#для решения и создания функции необходимо ззагрузить следующие библиотеки и функции
from sklearn.model_selection import train_test_split #функция понадобится для разделения данных на тренировочные и тестовые выборки
from sklearn.linear_model import LinearRegression #функция необходима для создания модели линейной регрессии
from sklearn.metrics import root_mean_squared_error #функция понадобится для вычисления метрики RMSE (среднеквадратичное отклонение) для оценки качества предсказаний модели и "глупой" модели.  
import pandas as pd #библиотека понадобится для работы с файлом, а именно открытие
import numpy as np #понадобится для применения метода для создания массива для "глупой модели"

def get(name_file): #Реализуется вся логика программы в функции, которая в качестве аргумента принимает строку - имя файла.

  file = pd.read_csv(name_file) #выбираем в качестве аргумента имя файла
  
  #В качестве предикторов (features) следующие столбцы: Year, Engine HP, Engine Cylinders, highway MPG, city mpg и Popularity. Целевым признаком (target) будет столбец MSRP.
  features = file[['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity']]
  target = file['MSRP']
  features = features.fillna(features.mean()) #Обработка пропущенных значений в признаках: замена их на средние значения соответствующих столбцов с помощью метода fillna.
  features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=42) #Разделиние данных на тренировочную и тестовую выборки в пропорции 75/25, используя функцию train_test_split из библиотеки sklearn.

  model = LinearRegression() #Создание модели линейной регрессии с помощью класса LinearRegression
  model.fit(features_train, target_train) #Обучение модели на тренировочной выборке, вызвав метод fit.
  r = model.predict(features_test) #Выполнение предсказания на тестовой выборке, используя метод predict.

  rmse = root_mean_squared_error(target_test, r) #Вычисление метрики RMSE (среднеквадратичное отклонение) для оценки качества предсказаний модели. Для этого используем функцию root_mean_squared_error из библиотеки sklearn.

#Реализация "глупую модель", которая для всех примеров предсказывает одно и то же значение — среднее значение целевого признака из тренировочной выборки.
  stupid = target_train.mean()
  dumb_model = np.full(len(target_test), target_train.mean()) #массив одинаковых значений длиной как test.
  drmse = root_mean_squared_error(target_test, dumb_model) #Вычисление RMSE для "глупой модели"

  return (rmse, drmse, drmse - rmse) #вывод RMSE линейной регрессии с RMSE "глупой модели", а также разница между ними
