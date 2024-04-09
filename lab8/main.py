import numpy as np

print('2.')
# Оцінки студента за семестр
grades = [85, 90, 75, 60, 92, 80]
# Створення одновимірного масиву numpy
grades_array = np.array(grades)
print("Оцінки студента у форматі масиву numpy:", grades_array)
print('\n')


print('3.')
# Список зі списками, що описують ціни на продукти в різних магазинах
prices = [
    [2.5, 1.2, 1.8, 4.5],  # ціни в першому магазині
    [2.8, 1.0, 1.6, 4.8],  # ціни в другому магазині
    [2.6, 1.4, 2.0, 4.3],  # ціни в третьому магазині
]
# Створення матриці numpy зі списків цін
prices_matrix = np.array(prices)

print("Матриця цін в магазинах:")
print(prices_matrix)
print('\n')


print('4.')
print("Тип даних значень з масиву із завдання 2:", grades_array.dtype)
print("Тип даних значень з масиву із завдання 3:", prices_matrix.dtype)
print('\n')


print('5.')
print("Форма масиву з завдання 2:", grades_array.shape)
print("Форма масиву з завдання 3:", prices_matrix.shape)
print('\n')


print('6.')
print('Пояснення:\tОтримання форми (n,) для масиву з завдання 2 означає, що це одновимірний масив (вектор) з n елементами. \n\tДля створення вектора-стовбця з цього масиву можна використати метод reshape(-1, 1), \n\tякий перетворить його у вектор-стовбець з одним стовбцем.')
grades_column_vector = grades_array.reshape(-1, 1)

print("Вектор-стовбець з оцінками:")
print(grades_column_vector)
print("Форма вектора-стовбця:", grades_column_vector.shape)
print('\n')


print('7.')
print('\t1. NumPy масиви швидше та ефективніше, особливо для числових обчислень.\n\t2. Вони займають менше пам\'яті, що робить їх ефективними для обробки великих об\'ємів даних.\n\t3. Масиви дозволяють виконувати операції над цілими наборами даних одночасно, уникнувши циклів.\n\t4. NumPy масиви мають статично типізований інтерфейс, тобто кожен елемент масиву має однаковий тип даних.')
print('\n')

print('8.')
profit = np.arange(0, 1000.51, 20.01)
print(profit)
print('\n')


print('9.')
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

print('Масив 1:')
print(array1)
print('Масив 2:')
print(array2)

print("Вертикальне об'єднання:")
print(np.vstack((array1, array2)))
print("Горизонтальне об'єднання")
print(np.hstack((array1, array2)))
print('\n')


print('10.')
def transpose_array(arr):
    """
    Transposes the input array using the reshape method.

    Args:
        arr (numpy.ndarray): The input array to be transposed.

    Returns:
        numpy.ndarray: The transposed array.
    """
    # Reshape the array with the dimensions reversed
    transposed_arr = arr.reshape(arr.shape[::-1])
    return transposed_arr

# Example usage:
input_array = np.array([[0, 1, 2, 3],
                        [4, 5, 6, 7]])

transposed_array = transpose_array(input_array)
print("Original Array:")
print(input_array)
print("\nTransposed Array:")
print(transposed_array)

print('\n')


print('11.')
# Create two arrays
array1 = np.array([1, 2])
array2 = np.array([3, 4])
print('Масив 1:')
print(array1)
print('Масив 2:')
print(array2)

# 1) Element-wise addition
elementwise_addition = array1 + array2

# 2) Element-wise subtraction
elementwise_subtraction = array1 - array2

# 3) Multiplication of an array by a scalar
scalar = 2
array_multiplication_scalar = scalar * array1

# 4) Element-wise multiplication
elementwise_multiplication = array1 * array2

# 5) Matrix multiplication
matrix_multiplication = np.dot(array1, array2)

# Printing the results
print("\n1) Element-wise addition:")
print(elementwise_addition)
print("\n2) Element-wise subtraction:")
print(elementwise_subtraction)
print("\n3) Multiplication of array by a scalar ([1, 2] * 2):")
print(array_multiplication_scalar)
print("\n4) Element-wise multiplication:")
print(elementwise_multiplication)
print("\n5) Matrix multiplication:")
print(matrix_multiplication)
print('\n')


print('12.')
matrix = np.array([[1, 2, 3, 4, 5],
                   [4, 5, 6, 7, 8],
                   [7, 8, 9, 10, 11],
                   [11, 12, 13, 14, 15],
                   [15, 16, 17, 18, 19]])
print('Матриця:')
print(matrix)

# 1) Minimum value
min_value = np.min(matrix)

# 2) Maximum value
max_value = np.max(matrix)

# 3) Sum of all elements
sum_of_elements = np.sum(matrix)

# 4) Minimum values for each row
min_values_per_row = np.min(matrix, axis=1)

# 5) Maximum values for each column
max_values_per_column = np.max(matrix, axis=0)

# Printing the results
print("1) Minimum value:", min_value)
print("2) Maximum value:", max_value)
print("3) Sum of all elements:", sum_of_elements)
print("4) Minimum values for each row:", min_values_per_row)
print("5) Maximum values for each column:", max_values_per_column)
print('\n')


print('13.')
# Getting values of the first and second columns for rows except the first and last
selected_values = matrix[1:-1, :2]

print("Selected values of the first and second columns for rows except the first and last:")
print(selected_values)
print('\n')


print('14.')
matrix = np.array([[1, 2, 3],
                   [4, 2, 6],
                   [1, 8, 9]])

unique_values, counts = np.unique(matrix, return_counts=True)

print("Unique values in the matrix:", unique_values)
print("Counts of unique values:    ", counts)
print('\n')


print('15.')
height = 480
width = 720

# Створення тривимірного масиву зображення
image = np.zeros((height, width, 3), dtype=np.uint8)

# Розділення зображення навпіл по висоті
middle_pixel = height // 2

# Розфарбовуємо верхню частину зображення в колір (0, 87, 184)
image[:middle_pixel, :, :] = [0, 87, 184]

# Розфарбовуємо нижню частину зображення в колір (255, 215, 0)
image[middle_pixel:, :, :] = [255, 215, 0]

# Виведення зображення
print(image)
print('\n')