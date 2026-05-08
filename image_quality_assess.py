import numpy as np
import math
import sys
from skimage import io, color, filters
import os
from tqdm import tqdm

def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    
    # ================= UCIQE =================
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]

    # 1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    # 2nd term
    top = np.int64(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top]) - np.mean(sl[:top])

    # 3rd term (Optimized: 完全向量化取代 for 循环)
    satur = np.zeros_like(chroma)
    # 创建掩码：排除 l == 0 或 chroma == 0 的情况
    valid_mask = (l != 0) & (chroma != 0)
    satur[valid_mask] = chroma[valid_mask] / l[valid_mask]
    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # ================= UIQM =================
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    # 1st term UICM
    rg = rgb[:,:,0].astype(np.float32) - rgb[:,:,1].astype(np.float32)
    yb = (rgb[:,:,0].astype(np.float32) + rgb[:,:,1].astype(np.float32)) / 2 - rgb[:,:,2].astype(np.float32)
    rgl = np.sort(rg, axis=None)
    ybl = np.sort(yb, axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int64(al1 * len(rgl))
    T2 = np.int64(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm = -0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel = np.clip(np.round(Rsobel), 0, 255).astype(np.uint8)
    Gsobel = np.clip(np.round(Gsobel), 0, 255).astype(np.uint8)
    Bsobel = np.clip(np.round(Bsobel), 0, 255).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # 3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm

    # ================= Luminance =================
    # 直接复用已经读取进内存的 rgb 数组，避免二次 IO 和报错
    img_float = rgb.astype(np.float32)
    weights = np.array([0.2126, 0.7152, 0.0722])
    # 注意 skimage io.imread 默认读取就是 RGB 顺序，所以可以直接点乘
    luminance_map = np.dot(img_float, weights)
    luminance = np.mean(luminance_map)
    
    return uiqm, uciqe, luminance

def eme(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):
        xlb = i * blocksize
        xrb = min((i + 1) * blocksize, ch.shape[0])

        for j in range(num_y):
            ylb = j * blocksize
            yrb = min((j + 1) * blocksize, ch.shape[1])
            
            block = ch[xlb:xrb, ylb:yrb]
            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            if blockmin == 0: blockmin += 1
            if blockmax == 0: blockmax += 1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i, j, gamma=1026):
    return i + j - i * j / gamma

def plipsub(i, j, k=1026):
    return k * (i - j) / (k - j)

def plipmult(c, j, gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):
        xlb = i * blocksize
        xrb = min((i + 1) * blocksize, ch.shape[0])

        for j in range(num_y):
            ylb = j * blocksize
            yrb = min((j + 1) * blocksize, ch.shape[1])
            
            block = ch[xlb:xrb, ylb:yrb]
            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            # 防止除零警告
            if bottom == 0:
                s += 0
            else:
                m = top / bottom
                if m > 0:
                    s += m * np.log(m)

    return plipmult(w, s)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_images>")
        sys.exit(1)

    result_path = sys.argv[1]
    result_dirs = os.listdir(result_path)

    sumuiqm, sumuciqe, sumluminance = 0., 0., 0.
    N = 0
    
    for imgdir in tqdm(result_dirs):
        # 修复逻辑漏洞：正确筛选图像文件
        if imgdir.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            try:
                corrected = io.imread(os.path.join(result_path, imgdir))
                
                # 如果有 Alpha 通道 (RGBA)，只取前三个通道
                if corrected.shape[-1] == 4:
                    corrected = corrected[..., :3]

                uiqm, uciqe, luminance = nmetrics(corrected)
                # tqdm 会管理打印，不会打乱进度条
                tqdm.write(f"[{imgdir}] UIQM: {uiqm:.4f}, UCIQE: {uciqe:.4f}, Luminance: {luminance:.4f}")

                sumuiqm += uiqm
                sumuciqe += uciqe
                sumluminance += luminance
                N += 1

                with open(os.path.join(result_path, 'metrics.txt'), 'a') as f:
                    f.write('{}: uiqm={:.4f} uciqe={:.4f} luminance={:.4f}\n'.format(imgdir, uiqm, uciqe, luminance))
            except Exception as e:
                tqdm.write(f"Error processing {imgdir}: {e}")

    if N > 0:
        muiqm = sumuiqm / N
        muciqe = sumuciqe / N
        mluminance = sumluminance / N

        with open(os.path.join(result_path, 'metrics.txt'), 'a') as f:
            f.write('\n=== Average ===\n')
            f.write('UIQM: {:.4f}\nUCIQE: {:.4f}\nLuminance: {:.4f}\n'.format(muiqm, muciqe, mluminance))
            
        print('\n=== Final Average ===')
        print(f'UIQM: {muiqm:.4f}')
        print(f'UCIQE: {muciqe:.4f}')
        print(f'Luminance: {mluminance:.4f}')

if __name__ == '__main__':
    main()