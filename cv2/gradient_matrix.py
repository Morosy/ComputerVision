import numpy as np
import csv
import cv2

# CSVから画像を読み込み
def read_csv_image(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        return np.array([[int(value.strip()) for value in row] for row in reader], dtype=np.float32)

# 指定された赤い画素の座標 (x, y)
x, y = 3, 1  # 例: 中央の画素を指定

# 画像を読み込み
img = read_csv_image('input/koubai_input.csv')

# Sobelフィルタによる勾配（OpenCVはデフォルトで x: 水平方向, y: 垂直方向）
sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)

# 勾配ベクトル
Ix = sobel_x[y, x]
Iy = sobel_y[y, x]

# Structure tensor（外積）
structure_tensor = np.array([
    [Ix * Ix, Ix * Iy],
    [Ix * Iy, Iy * Iy]
])

# 結果出力
print(f"座標 (x={x}, y={y}) におけるSobel勾配:")
print(f"Ix = {Ix}")
print(f"Iy = {Iy}")
print("\nStructure tensor ∇I ∇Iᵀ:")
print(structure_tensor)
