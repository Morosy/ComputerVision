import numpy as np
import csv

# CSV読み込み関数
def read_csv_image(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        return np.array([[float(value.strip()) for value in row] for row in reader], dtype=np.float32)

# SADによるテンプレートマッチング
def compute_sad_map(img, template):
    H, W = img.shape
    h, w = template.shape
    sad_map = np.zeros((H - h + 1, W - w + 1), dtype=int)

    for i in range(H - h + 1):
        for j in range(W - w + 1):
            patch = img[i:i+h, j:j+w]
            sad = np.sum(np.abs(patch - template))
            sad_map[i, j] = sad

    return sad_map

# 入力画像とテンプレートを読み込む
img = read_csv_image('input/template_matching_img.csv')
template = read_csv_image('input/template_matching_template.csv')

# SAD相違度画像の計算
sad_result = compute_sad_map(img, template)

# 結果表示
print("相違度画像 (SAD):")
print(sad_result)
