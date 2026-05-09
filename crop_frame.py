import os
import cv2
import multiprocessing
import time
import sys

def extract_frames(video_file, output_dir, name_prefix):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_filename = os.path.join(
            output_dir, f"{name_prefix}_{frame_count:06d}.png"
        )
        cv2.imwrite(frame_filename, frame)
    cap.release()

def process_video(video_file):
    video_stem = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = sys.argv[2]
    extract_frames(video_file, output_dir, video_stem)

if __name__ == "__main__":
    start = time.time()
    input_folder = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".mp4")]
    pool = multiprocessing.Pool(processes=12)
    pool.map(process_video, video_files)
    pool.close()
    pool.join()
    end = time.time()
    print('time: ',end - start)