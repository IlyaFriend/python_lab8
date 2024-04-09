import pandas as pd
import numpy as np
from datetime import datetime



print('2.')
data = {
    'Прізвище': ['Іванов', 'Петров', 'Сидоров', 'Коваленко', 'Михайленко'],
    'Ім\'я': ['Іван', 'Петро', 'Олег', 'Марина', 'Анна'],
    'Дата народження': ['1980-01-01', '1990-05-15', '1985-10-20', '1975-03-08', '2000-12-30'],
    'Маса тіла (кг)': [75, 82, 68, None, 65],
    'Медичне страхування': [True, True, False, True, False]
}

df = pd.DataFrame(data)

print(df.dtypes)
print('\n')


print('3.')
df = pd.read_csv("missile_attacks_daily.csv")

print(df)
print('\n')


print('4.')
print(df.head(10))
print('\n')


print('5.')
print(df.shape)
print('\n')

print('6.')
print(df.info())
print('\n')


print('7.')
print(df.describe())
print('\n')

print('8.')
print(df.nunique())
print('\n')


print('9.')
min_unique_column = df.nunique().idxmin()
value_counts = df[min_unique_column].value_counts()
print(value_counts)
print('\n')


print('10.')
print('\n\t1.Деякі колонки можуть мати неправильні типи даних. особливо, наприклад, дату можна перетворити на тип datetime, якщо вона зберігається як рядок.\n\t2.Потрібно вивчити рядки з відсутніми даними і вирішити, яким чином їх заповнити або видалити.\n\t3.Якщо датасет містить числові дані, які мають різні масштаби, можна виконати їх стандартизацію\n\t')
print('\n')


print('11.')
print(df['launched'].head(5))
df = df.dropna(subset=['launched'])
df = df[~df['launched'].isin([np.nan, np.inf, -np.inf])]

# Перетворення колонки 'launched' у цілочисельний тип
df['launched'] = df['launched'].astype('int64')
print('_________________')
print(df['launched'].head(5))
print('\n')


print('12.')
# Створення копії DataFrame
df_copy = df.copy()

print('До видалення', df_copy.shape)

# Розрахунок відсотка непустих значень для кожної колонки
non_null_percentages = df_copy.count() / len(df_copy)

# Вибір колонок, де відсоток непустих значень менше 30%
columns_to_drop = non_null_percentages[non_null_percentages < 0.3].index

# Видалення відповідних колонок
df_copy = df_copy.drop(columns=columns_to_drop)

print('Після видалення', df_copy.shape)
print('\n')


print('13.')
print('До видалення', df.shape)
# Збереження початкового датафрейму для порівняння
df_before_cleaning = df.copy()


# Функція для перевірки формату дати
def is_valid_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d %H:%M')
        return True
    except ValueError:
        return False

# Функція для приведення дат в один формат у межах рядка
def fix_date(date_str, end_time_col):
    try:
        pd.to_datetime(date_str, format='%Y-%m-%d %H:%M')
        return date_str
    except ValueError:
        if len(str(end_time_col)) >= 16:  # Перевірка формату time_end на наявність часу
            end_time = str(end_time_col)[-5:]  # Витягуємо час з колонки time_end
            return date_str + ' ' + end_time
        else:
            return date_str

# Зчитуємо файли та перевіряємо формат дати
df = pd.read_csv("missile_attacks_daily.csv")

# Виправлення значень колонки time_start
df['time_start'] = df.apply(lambda row: fix_date(row['time_start'], row['time_end']), axis=1)

# Видалення записів, де time_end пізніший за time_start
df = df.dropna(subset=['time_start'])

# Зчитуємо файли та перевіряємо формат дати
# df = pd.read_csv("missile_attacks_daily.csv")

# Виправлення значень колонки time_start
# df['time_start'] = df.apply(lambda row: fix_date(row['time_start'], row['time_end']), axis=1)

df_before_cleaning = df.copy()
# Перевірка та видалення записів для дат з форматом '%Y-%m-%d'
date_format = '%Y-%m-%d'
mask = (pd.to_datetime(df['time_start'], errors='coerce', format=date_format) > pd.to_datetime(df['time_end'], errors='coerce', format=date_format))
df = df[~mask]

# Перевірка та видалення записів для дат з форматом '%Y-%m-%d %H:%M'
date_format = '%Y-%m-%d %H:%M'
mask = (pd.to_datetime(df['time_start'], errors='coerce', format=date_format) > pd.to_datetime(df['time_end'], errors='coerce', format=date_format))
df = df[~mask]

print(df[['time_start', 'time_end']].head(15))

# Оновлення індексу
df.reset_index(drop=True, inplace=True)

# Кількість видалених рядків
deleted_rows_count = len(df_before_cleaning) - len(df)
print('Після видалення', df.shape)
print('rows deleted', deleted_rows_count)
merged_df = df_before_cleaning.merge(df, how='left', indicator=True)

# Вибираємо записи, які є у першому датафреймі, але відсутні в другому
unique_to_old = merged_df[merged_df['_merge'] == 'left_only']

# Виводимо ці записи
print('Рядки, де дата старту більша за дату кінця', unique_to_old)
print('\n')


print('14.')
rows_with_missing_values = df[df.isnull().any(axis=1)]
print(rows_with_missing_values)
print('\n')


print('15.')
# Вибір рядків, що відповідають моделям ракет зі списку
selected_models = ['X-31', 'Kalibr', 'Iskander-M', 'X-59', 'X-101/X-555', 'X-101', 'X-555', 'X-31P', 'X-22', 'X-47', 'X-59 and X-35', 'X-47 Kinzhal', 'Iskander-M/KN-23/X-47', 'KN-23']
selected_rows = df[df['model'].isin(selected_models)]

# Обчислення загальної кількості запущених і знищених ракет
total_launched = selected_rows['launched'].sum()
total_destroyed = selected_rows['destroyed'].sum()

# Розрахунок відсотка збиття
if total_launched != 0:
    average_percentage_destroyed = (total_destroyed / total_launched) * 100
else:
    average_percentage_destroyed = 0

print("Середній відсоток збиття ракет за весь наведений період: {:.2f}%".format(average_percentage_destroyed))
print('\n')

print('16.')
# Згрупувати дані за моделлю та підрахувати загальну кількість випущених пристроїв
model_counts = df.groupby('model')['launched'].sum().reset_index()

# Відсортувати дані у зворотньому порядку за кількістю випущених пристроїв
top_10_models = model_counts.sort_values(by='launched', ascending=False).head(10)

# Перетворити дані у формат JSON і зберегти у файл
top_10_models.to_json('top_10_models.json', orient='records')

# Вивести перші 10 найпопулярніших моделей запущених пристроїв
print(top_10_models)
print('\n')

print('17.')
# Створюємо нову колонку, яка буде містити різницю між кількістю запущених та збитих пристроїв
df['hits'] = df['launched'] - df['destroyed']

# Групуємо дані за датою початку атаки і рахуємо сумарну кількість влучань (різниця між запущеними та збитими) для кожного дня
hits_by_day = df.groupby('time_start')['hits'].sum()

# Знаходимо день з найбільшою кількістю влучань
max_hits_day = hits_by_day.idxmax()

# Отримуємо кількість влучань у цей день
max_hits_count = hits_by_day[max_hits_day]

# Виводимо час початку атаки та кількість влучань (різниця між запущеними та збитими) для дня з найбільшою кількістю влучань
print("Найбільше влучань було у день:", max_hits_day)
print("Кількість влучань (різниця між запущеними та збитими):", max_hits_count)
print('\n')

print('18.')
# Фільтрація датафрейму за форматом дати
df_valid_dates = df[df['time_start'].apply(is_valid_date)]

# Розділення дати на рік та місяць
df_valid_dates['year'] = pd.to_datetime(df_valid_dates['time_start']).dt.year
df_valid_dates['month'] = pd.to_datetime(df_valid_dates['time_start']).dt.month

# Розрахунок тривалості атаки
df_valid_dates['attack_duration'] = pd.to_datetime(df_valid_dates['time_end']) - pd.to_datetime(df_valid_dates['time_start'])

# Знаходження найбільш тривалої та найбільш короткої атаки за кожен місяць
result = df_valid_dates.groupby(['year', 'month'])['attack_duration'].agg(['min', 'max']).reset_index()
result.columns = ['year', 'month', 'shortest_attack', 'longest_attack']

# Виведення результату
# Записати результати у файл CSV
result.to_csv('attack_durations_by_month.csv', index=False)

# Вивести результати на екран
print(result)