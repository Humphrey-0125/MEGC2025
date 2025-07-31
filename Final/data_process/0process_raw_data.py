import os
import pandas as pd
import cv2
import numpy as np

# 读取 Excel 文件
df = pd.read_excel('/home/d1/zwb/yhf/CAS(ME)II/CASME2-coding-20140508.xlsx')

# 视频帧所在的源文件夹
video_frames_folder = '/home/d1/zwb/yhf/CAS(ME)II/CASME2_RAW_selected'

# 输出文件夹（保存提取的视频帧）
output_root_folder = '/home/d1/zwb/yhf/CAS(ME)II/output_zero'

# 遍历每一行数据并提取视频帧
for index, row in df.iterrows():
    # 提取信息
    subject = str(row['Subject']).zfill(2)  # 确保 Subject 是两位数
    filename = row['Filename']
    onset_frame = int(row['OnsetFrame'])  # 确保 Onset Frame 是整数
    offset_frame = int(row['OffsetFrame'])  # 确保 Offset Frame 是整数

    subject = f"sub{subject}"

    # 构建视频文件夹路径
    video_folder = os.path.join(video_frames_folder, subject, filename)

    # 构建输出文件夹路径
    output_folder = os.path.join(output_root_folder, subject, filename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 计算总帧数
    total_frames = offset_frame - onset_frame + 1

    # 均匀采样 10 帧
    if total_frames >= 10:
        # 使用 numpy.linspace 来生成 10 个等间隔的帧
        sampled_frames = np.linspace(onset_frame, offset_frame, 10, dtype=int)
    else:
        # 如果帧数小于10，就直接使用所有帧
        sampled_frames = list(range(onset_frame, offset_frame + 1))

    # 遍历采样帧并提取图像
    for frame_num in sampled_frames:
        # 构建视频帧文件路径
        frame_filename = f"img{frame_num}.jpg"
        frame_path = os.path.join(video_folder, frame_filename)

        # 检查该帧是否存在
        if os.path.exists(frame_path):
            # 读取帧
            frame = cv2.imread(frame_path)

            # 构建输出路径
            output_frame_path = os.path.join(output_folder, f"{frame_filename}")

            # 保存帧到输出文件夹
            cv2.imwrite(output_frame_path, frame)
        else:
            print(f"Warning: {frame_path} does not exist, skipping.")

print("Frames extraction completed!")
