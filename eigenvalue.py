import numpy as np
import csv

# CSV から行列を読み込む関数
def read_matrix_from_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        matrix = [[float(value.strip()) for value in row] for row in reader]
    return np.array(matrix)

# 行列の読み込み
A = read_matrix_from_csv('input/eigenvalue_input.csv')

# 固有値の計算
eigenvalues = np.linalg.eigvals(A)
lambda1, lambda2 = eigenvalues

# 固有値の和と積
sum_eigen = lambda1 + lambda2
product_eigen = lambda1 * lambda2

# 結果の表示
print("行列 A:")
print(A)
print("固有値:", eigenvalues)
print("固有値の和 (λ1 + λ2):", sum_eigen)
print("固有値の積 (λ1 * λ2):", product_eigen)
print("det(A):", np.linalg.det(A))
print("tr(A):", np.trace(A))

