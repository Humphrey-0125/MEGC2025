import cv2
import numpy as np
import os

import dlib  # 用于精确鼻尖定位
# 初始化dlib人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

import pandas as pd
# 读取 Excel 文件
df = pd.read_excel('/home/d1/zwb/yhf/CAS(ME)II/CASME2-coding-20140508.xlsx')

# 视频帧所在的源文件夹
image_sequences_root = '/home/d1/zwb/yhf/CAS(ME)II/CASME2_RAW_selected'

# 输出文件夹（保存提取的视频帧）
output_root = '/home/d1/zwb/yhf/CAS(ME)II/output'
def get_nose_region(image):
    """精确鼻尖区域定位（基于dlib 68点模型）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        # 若未检测到人脸，使用简化中心区域
        height, width = gray.shape[:2]
        return slice(height//2-20, height//2+20), slice(width//2-15, width//2+15)
    
    # 获取68个面部关键点
    landmarks = predictor(gray, faces[0])
    # 鼻尖关键点 = 索引30（OpenCV 68点模型标准）
    nose_tip_x = landmarks.part(30).x
    nose_tip_y = landmarks.part(30).y
    
    # 定义鼻尖ROI区域 (40x30像素？？？)
    y_start = max(0, nose_tip_y - 20)
    y_end = min(gray.shape[0], nose_tip_y + 20)
    x_start = max(0, nose_tip_x - 15)
    x_end = min(gray.shape[1], nose_tip_x + 15)
    
    return slice(y_start, y_end), slice(x_start, x_end)

# def load_image_sequence(folder):
#     """按数字顺序加载以数字命名的图像序列"""
#     # 获取文件夹中的所有图像文件
#     filenames = [
#         f for f in os.listdir(folder)
#         if f.lower().endswith((".jpg", ".png", ".bmp"))
#     ]
#     # 过滤出纯数字文件名的图像
#     numeric_filenames = []
#     for f in filenames:
#         # 去除扩展名后的文件名部分
#         name_without_ext = os.path.splitext(f)[0]
#         # 检查是否完全由数字组成
#         if name_without_ext.isdigit():
#             numeric_filenames.append(f)
#     # 加载所有图像
#     numeric_filenames.sort(key=lambda f: int(os.path.splitext(f)[0]))
#     frames = [cv2.imread(os.path.join(folder, f)) for f in numeric_filenames]
#     return frames, numeric_filenames

# def find_apex_frame_index(frames):
#     """使用差分最大法找到 apex 帧索引"""
#     onset_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
#     max_diff = 0
#     apex_idx = 0
#     for i in range(1, len(frames)):
#         current_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
#         diff = np.sum(cv2.absdiff(onset_gray, current_gray))
#         if diff > max_diff:
#             max_diff = diff
#             apex_idx = i
#     return apex_idx

def compute_optical_flow(onset_frame, apex_frame):
    # 转换为灰度图
    onset_gray = cv2.cvtColor(onset_frame, cv2.COLOR_BGR2GRAY)
    apex_gray = cv2.cvtColor(apex_frame, cv2.COLOR_BGR2GRAY)
    
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
        tau=0.25,       
        lambda_=0.15,   
        theta=0.3,      
        nscales=1,      
        warps=5         
    )
    # flow = tvl1.calc(onset_gray, apex_gray, None)
    # return flow

    # 鼻尖区域运动补偿
    flow = tvl1.calc(onset_gray, apex_gray, None)
    nose_roi = get_nose_region(onset_frame)
    mean_nose_flow = np.mean(flow[nose_roi], axis=(0, 1))
    # 全局补偿：减去鼻尖平均位移
    compensated_flow = flow - mean_nose_flow
    return compensated_flow
    

def flow_to_color_map(flow, onset_gray):
    # 计算运动幅度和角度
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hue = angle * 180 / np.pi / 2
    # 幅度→明度映射（归一化到0-255）
    value = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # 构建HSV图像（饱和度固定最大值）
    saturation = 255 * np.ones_like(hue)
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.uint8)
    # 转换为BGR色彩空间
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 叠加到起始帧灰度图   这个权重比例怎样设置？？ 暂时5:5
    onset_bgr = cv2.cvtColor(onset_gray, cv2.COLOR_GRAY2BGR)
    mmc_map = cv2.addWeighted(onset_bgr, 0.5, flow_color, 0.5, 0)
    return mmc_map


def generate_mmc_from_excel_info():
    # 确保输出根目录存在
    os.makedirs(output_root, exist_ok=True)
    # 遍历Excel中的每一行
    for index, row in df.iterrows():
        try:
            subject = str(row['Subject']).zfill(2)  # 确保是2位数字
            video_name = str(row['Filename'])
            onset_frame_num = int(row['OnsetFrame'])
            apex_frame_num = int(row['ApexFrame'])
            print(f"\n处理序列: Subject {subject}, Video {video_name}")
            print(f"Onset Frame: {onset_frame_num}, Apex Frame: {apex_frame_num}")
            subject = f"sub{subject}"
            # 构建图像序列文件夹路径 输出文件夹路径
            sequence_folder = os.path.join(image_sequences_root, subject, video_name)
            output_folder = os.path.join(output_root, subject, video_name)
            os.makedirs(output_folder, exist_ok=True)
            
            # 帧文件名（格式为img{帧号}.jpg）
            onset_filename = f"img{onset_frame_num}.jpg"
            apex_filename = f"img{apex_frame_num}.jpg"
            
            onset_path = os.path.join(sequence_folder, onset_filename)
            apex_path = os.path.join(sequence_folder, apex_filename)
            
            if not os.path.exists(onset_path):
                print(f"警告: Onset帧不存在 {onset_path}")
                continue
            if not os.path.exists(apex_path):
                print(f"警告: Apex帧不存在 {apex_path}")
                continue
            onset_frame = cv2.imread(onset_path)
            apex_frame = cv2.imread(apex_path)
            
            if onset_frame is None or apex_frame is None:
                print(f"错误: 无法读取图像 ({onset_path} 或 {apex_path})")
                continue
                
            # 计算光流
            flow = compute_optical_flow(onset_frame, apex_frame)

            onset_gray = cv2.cvtColor(onset_frame, cv2.COLOR_BGR2GRAY)
            mmc_map = flow_to_color_map(flow, onset_gray)
            
            output_path = os.path.join(output_folder, f"{subject}_{video_name}_mmc_map_no_nose.png")
            cv2.imwrite(output_path, mmc_map)
            print(f"MMC-Map 已保存至: {output_path}")
            
        except Exception as e:
            print(f"处理序列 {subject}/{video_name} 时出错: {str(e)}")
            # 处理序列 04/EP12_01f 时出错：invalid literal for int(） with base 10:'/'  原因是excel中未提供apex帧

if __name__ == "__main__":
    generate_mmc_from_excel_info()
    print("\n所有序列处理完成！")


# if __name__ == "__main__":
#     parent_folder = "/home/d1/zwb/yhf/MEGC_unseen_testset/ME_VQA_MEGC_2025_Test"
#     generate_mmc_for_all_sequences(parent_folder)
#     generate_mul_flow_maps_for_all_sequences(parent_folder)


