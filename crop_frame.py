import os
import cv2
import multiprocessing
import time
import sys
from tqdm import tqdm

def extract_frames(video_file, output_dir):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()

def process_video(video_file):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(sys.argv[2], video_name)
    os.makedirs(output_dir, exist_ok=True)
    extract_frames(video_file, output_dir)

if __name__ == "__main__":
    start = time.time()
    input_folder = sys.argv[1]
    video_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".mp4")
    ]
    if not video_files:
        print("未找到 .mp4 文件")
        sys.exit(1)
    with multiprocessing.Pool(processes=12) as pool:
        list(
            tqdm(
                pool.imap(process_video, video_files),
                total=len(video_files),
                desc="Extract frames",
                unit="video",
            )
        )
    end = time.time()
    print("time: ", end - start)