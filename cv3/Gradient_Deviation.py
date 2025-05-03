import numpy as np
import csv
import cv2
import math

# CSVから画像を読み込む関数
def read_csv_image(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        return np.array([[float(value.strip()) for value in row] for row in reader], dtype=np.float32)

# 画像読み込み
img = read_csv_image("input/cv3/Gradient_Deviation.csv")  # 例: グレースケール画像

# 赤い画素の位置（例: 中央 (x=1, y=1)）
x, y = 1, 2

# Sobelフィルタで勾配成分計算
sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)  # Ix
sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)  # Iy

Ix = sobel_x[y, x]
Iy = sobel_y[y, x]

# 勾配強度の2乗
Ix2 = Ix ** 2
Iy2 = Iy ** 2

# 偏角θ（ラジアンと度の両方）
theta_rad = math.atan2(Iy, Ix)
theta_deg = math.degrees(theta_rad)

# 出力
print(f"座標 (x={x}, y={y}) における結果：")
print(f"Ix = {Ix:.3f}")
print(f"Iy = {Iy:.3f}")
print(f"Ix^2 = {Ix2:.3f}")
print(f"Iy^2 = {Iy2:.3f}")
print(f"Ix^2 + Iy^2 = {Ix2 + Iy2:.3f}")
print(f"θ (radian) = {theta_rad:.3f}")
print(f"θ (degree) = {theta_deg:.2f}°")
