import numpy as np
import csv

# CSV読み込み関数
def read_csv_image(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        return np.array([[float(value.strip()) for value in row] for row in reader], dtype=np.float32)

# 入力画像の読み込み
img1 = read_csv_image('input/SADandSSDandNAC_1.csv')
img2 = read_csv_image('input/SADandSSDandNAC_2.csv')

# SAD
sad = np.sum(np.abs(img1 - img2))

# SSD
ssd = np.sum((img1 - img2) ** 2)

# NCC
numerator = np.sum(img1 * img2)
denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2))
ncc = numerator / denominator if denominator != 0 else 0

# 出力
print(f"SAD: {sad}")
print(f"SSD: {ssd}")
print(f"NCC 分子: {numerator}")
print(f"NCC 分母: {denominator}")
print(f"NCC: {ncc:.4f}")
