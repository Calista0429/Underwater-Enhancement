import cv2
import numpy as np
import math
import os
import csv
import sys
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ================= 核心计算函数 (保持不变) =================

def calculate_luminance(image):
    b, g, r = cv2.split(image)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return np.mean(luminance)

def calculate_uciqe(image):
    img_float = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    chroma = np.sqrt(np.square(a) + np.square(b))
    saturation = chroma / (L + 1e-6)
    sigma_c = np.std(chroma)
    mu_s = np.mean(saturation)
    L_flat = np.sort(L.flatten())
    idx_1_percent = max(1, int(len(L_flat) * 0.01))
    con_l = np.mean(L_flat[-idx_1_percent:]) - np.mean(L_flat[:idx_1_percent])
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s

def calculate_uicm(image):
    b, g, r = cv2.split(image.astype(np.float32))
    RG = r - g
    YB = 0.5 * (r + g) - b
    uicm = -0.0268 * np.sqrt(np.mean(RG)**2 + np.mean(YB)**2) + 0.1586 * np.sqrt(np.std(RG)**2 + np.std(YB)**2)
    return uicm

def get_eme(channel, window_size=8):
    h, w = channel.shape
    eme, num_blocks = 0.0, 0
    for i in range(0, h - window_size + 1, window_size):
        for j in range(0, w - window_size + 1, window_size):
            block = channel[i:i+window_size, j:j+window_size]
            b_min, b_max = np.min(block), np.max(block)
            if b_min > 0: eme += 20 * math.log(b_max / b_min)
            num_blocks += 1
    return eme / num_blocks if num_blocks > 0 else 0

def calculate_uiqm(image):
    uicm = calculate_uicm(image)
    b, g, r = cv2.split(image.astype(np.float32))
    uism = 0.299 * get_eme(r) + 0.587 * get_eme(g) + 0.114 * get_eme(b)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    uiconm = get_eme(gray)
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm

# ================= 文件夹处理与保存逻辑 =================

def batch_process(input_folder):
    # 1. 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误：文件夹 '{input_folder}' 不存在")
        return

    output_filename = os.path.join(input_folder, "metrics.csv")

    # 2. 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print("文件夹中没有找到有效的图片文件。")
        return

    print(f"开始处理，共 {len(image_files)} 张图片...")
    # 3. 循环计算并实时写入 CSV
    keys = ["File Name", "Luminance", "UCIQE", "UIQM"]
    sum_lum, sum_uciqe, sum_uiqm, valid_count = 0.0, 0.0, 0.0, 0

    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        f.flush()

        for filename in tqdm(image_files, desc="评估进度", unit="img"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"跳过无法读取的图片: {filename}")
                continue

            lum = calculate_luminance(img)
            uciqe = calculate_uciqe(img)
            uiqm = calculate_uiqm(img)

            row = {
                "File Name": filename,
                "Luminance": round(lum, 4),
                "UCIQE": round(uciqe, 4),
                "UIQM": round(uiqm, 4)
            }
            dict_writer.writerow(row)
            f.flush()

            sum_lum += lum
            sum_uciqe += uciqe
            sum_uiqm += uiqm
            valid_count += 1

        # 4. 计算平均值并追加到 CSV
        if valid_count > 0:
            avg_lum = sum_lum / valid_count
            avg_uciqe = sum_uciqe / valid_count
            avg_uiqm = sum_uiqm / valid_count
            avg_row = {
                "File Name": "AVERAGE",
                "Luminance": round(avg_lum, 4),
                "UCIQE": round(avg_uciqe, 4),
                "UIQM": round(avg_uiqm, 4)
            }
            dict_writer.writerow(avg_row)
            f.flush()

    print("-" * 30)
    print(f"处理完成！结果已保存至: {output_filename}")
    if valid_count > 0:
        print(f"平均结果 -> Lum: {avg_lum:.4f}, UCIQE: {avg_uciqe:.4f}, UIQM: {avg_uiqm:.4f}")
    else:
        print("未能成功读取任何图片，未计算平均值。")

# ================= 运行 =================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python assess.py <target_folder>")
        sys.exit(1)

    target_folder = sys.argv[1]
    batch_process(target_folder)