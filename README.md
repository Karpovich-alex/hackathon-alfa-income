# Решение Хакатона "Построение модели дохода от Альфа-Банка"
## Описание решения
В основе решения лежит модель LightGBM, параметры которой были подобраны Optuna.
Данные были предобработаны. Nan заполнены 0 для числовых столбцов.
Для интерпретации используется SHAP.

## Запуск проекта
1. Настроить и активировать виртуальную среду
```commandline
python -m venv vevn
source ./venv/Scripts/activate
```
2. Установить зависимости:
```commandline
pip install -r requirements.txt
```
2. Запустить UI
```commandline
python -m streamlit run main.py
```
