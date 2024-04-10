import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


print('1.')
df = pd.read_csv('train.csv')
print(df.info())
print(df.head())
print('\n')

print('2.')
survived_counts = df['Survived'].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(survived_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
plt.title('Survival Rate')
plt.axis('equal')
plt.show()
print('\n')

print('3.')
# конвертуємо до числа
df['Survived'] = df['Survived'].astype(int)

survived_by_sex = df.groupby('Sex', as_index=False)['Survived'].mean()

plt.figure(figsize=(8, 6))
plt.bar(survived_by_sex['Sex'], survived_by_sex['Survived'], color=['pink', 'blue'])
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Sex')
plt.show()
# Більшість жінок вижила (більше 70%). Сереж чоловіків виживших було відсотково набагато менше - до 20%. 
print('\n')

print('4.')
#  підраховуємо кількості пропущених значень
missing_values = df.isnull().sum()
# впорядковуємо змінні за спаданням кількості пропущених значень
missing_values_sorted = missing_values.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
missing_values_sorted.plot(kind='bar', color='skyblue')
plt.title('Кількість пропущених значень за змінними (впорядковано за спаданням)')
plt.xlabel('Змінні')
plt.ylabel('Кількість пропущених значень')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Отримані результати показують, що колонка "Cabin" має найбільшу кількість пропущених значень, майже 700. Колонка "Age" також має значну кількість пропусків, близько 175. Кількість пропущених значень у колонці "Embarked" дуже мала, менше 10. Всі інші колонки мають нульову кількість пропущених значень.
print('\n')

print('5.')
# В цьому завданні необхідно знайти, як вік пасажирів і клас, яким вони подорожували, вплинули на ймовірність вижити (треба дослідити звʼязки між цими параметрами).
f, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=df, split=True, ax=ax)

ax.set_title('Pclass and Age Vs. Survived')
ax.set_yticks(range(0, 100, 10)) 
plt.show()
# Чим нижче клас, тим більше частка молодих людей
# На відміну від 2-го і 3-го класів, у 1-му класі більшість людей середнього віку вижили.
print('\n')

print('6.')
# Вибір значень віку без пропущених
# інакше, median_age має значення nan
age_values = df['Age'].dropna()

plt.figure(figsize=(10, 6))
plt.hist(age_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

mean_age = np.mean(age_values)
median_age = np.median(age_values)

plt.axvline(mean_age, color='red', linewidth=1, label='Середнє значення')
plt.axvline(median_age, color='green', linewidth=1, label='Медіана')

plt.title('Розподіл частот по віку пасажирів')
plt.xlabel('Вік')
plt.ylabel('Частота')

plt.legend()
plt.show()
# Більшість пасажирів мали вік 20-40 років, пасажиири, яким більше сорока років зменшуються
# Серед дітей найбільше було дітей 0-2 років 
# Середній і медіанний вік приблизно 30 років
print('\n')

print('7.')
# # Як місце (місто) посадки впинуло на шанс вижити.
survival_by_embarked = df.groupby('Embarked')['Survived'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Embarked', y='Survived', data=survival_by_embarked)
plt.title('Survival Rate by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Survival Rate')
plt.show()
# # Пасажирам, які сіли у Cherbourg'y, пощастило більше, ніж іншим, так як більшість із них вижила.
# # Пасажири, які сіли у Queenstownʼі та Southampton, у більшості загинули (виживші приблизно 40 і 35 відсотків відповідно)
print('\n')