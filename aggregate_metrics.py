import os
import sys
import glob

def calculate_global_average(main_folder):
    total_uiqm = 0.0
    total_uciqe = 0.0
    total_luminance = 0.0
    total_images = 0

    print(f"正在扫描路径: {main_folder} ...")
    
    # 匹配所有子文件夹下的 metrics.txt
    txt_files = glob.glob(os.path.join(main_folder, '*', 'metrics.txt'))

    if not txt_files:
        print(f"❌ 错误：未在 '{main_folder}' 的子文件夹中找到任何 metrics.txt 文件！")
        return

    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # 过滤掉无用的汇总行和空行
                if 'Average' in line or '===' in line or not line.strip():
                    continue
                
                # 提取图片指标
                if 'uiqm=' in line and 'uciqe=' in line:
                    try:
                        parts = line.strip().split()
                        uiqm_val = float(parts[1].split('=')[1])
                        uciqe_val = float(parts[2].split('=')[1])
                        lum_val = float(parts[3].split('=')[1])

                        total_uiqm += uiqm_val
                        total_uciqe += uciqe_val
                        total_luminance += lum_val
                        total_images += 1
                        
                    except Exception as e:
                        print(f"解析警告: 文件 {txt_file} 中的 '{line.strip()}' 无法解析 - {e}")

    # 计算最终的全局平均并保存
    if total_images > 0:
        global_uiqm = total_uiqm / total_images
        global_uciqe = total_uciqe / total_images
        global_luminance = total_luminance / total_images
        
        # 构建输出的文本内容
        summary_text = (
            f"{'='*40}\n"
            f"全局统计完成！共扫描了 {len(txt_files)} 个文件夹，包含 {total_images} 张图片。\n"
            f"{'-'*40}\n"
            f"【全局真实平均 UIQM】:      {global_uiqm:.4f}\n"
            f"【全局真实平均 UCIQE】:     {global_uciqe:.4f}\n"
            f"【全局真实平均 Luminance】: {global_luminance:.4f}\n"
            f"{'='*40}\n"
        )
        
        # 1. 打印到控制台
        print(summary_text)
        
        # 2. 保存到 txt 文件中
        output_file = os.path.join(main_folder, 'global_summary.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
            
        print(f"✅ 统计结果已成功保存至: {output_file}")
    else:
        print("⚠️ 解析完成，但没有提取到有效的单张图片数据。请检查 txt 文件的格式。")

if __name__ == '__main__':
    # 检查命令行参数是否足够
    if len(sys.argv) < 2:
        print("用法: python script.py <主文件夹的路径>")
        print("示例: python script.py /home/user/underwater_datasets")
        sys.exit(1)
        
    # 获取传入的路径
    TARGET_MAIN_FOLDER = sys.argv[1]
    
    # 检查路径是否存在
    if not os.path.isdir(TARGET_MAIN_FOLDER):
        print(f"❌ 错误：找不到文件夹 '{TARGET_MAIN_FOLDER}'。请检查路径是否拼写正确。")
        sys.exit(1)
        
    calculate_global_average(TARGET_MAIN_FOLDER)