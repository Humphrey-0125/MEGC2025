import cv2
import numpy as np
import os

import dlib  # 用于精确鼻尖定位
# 初始化dlib人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

def load_image_sequence(folder):
    """按数字顺序加载以数字命名的图像序列"""
    # 获取文件夹中的所有图像文件
    filenames = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".bmp"))
    ]
    # 过滤出纯数字文件名的图像
    numeric_filenames = []
    for f in filenames:
        # 去除扩展名后的文件名部分
        name_without_ext = os.path.splitext(f)[0]
        # 检查是否完全由数字组成
        if name_without_ext.isdigit():
            numeric_filenames.append(f)
    # 加载所有图像
    numeric_filenames.sort(key=lambda f: int(os.path.splitext(f)[0]))
    frames = [cv2.imread(os.path.join(folder, f)) for f in numeric_filenames]
    return frames, numeric_filenames

def find_apex_frame_index(frames):
    """使用差分最大法找到 apex 帧索引"""
    onset_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    max_diff = 0
    apex_idx = 0
    for i in range(1, len(frames)):
        current_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = np.sum(cv2.absdiff(onset_gray, current_gray))
        if diff > max_diff:
            max_diff = diff
            apex_idx = i
    return apex_idx

def compute_optical_flow(onset_frame, apex_frame):
    # 转换为灰度图
    onset_gray = cv2.cvtColor(onset_frame, cv2.COLOR_BGR2GRAY)
    apex_gray = cv2.cvtColor(apex_frame, cv2.COLOR_BGR2GRAY)
    
    # 使用TV-L1算法计算稠密光流（论文指定方法）
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
        tau=0.25,       # 时间步长（收敛速度）
        lambda_=0.15,   # 平滑项权重
        theta=0.3,      # 平衡系数
        nscales=1,      # 金字塔缩放层数
        warps=5         # 扭曲次数
    )
    # flow = tvl1.calc(onset_gray, apex_gray, None)
    # return flow

    # 鼻尖区域运动补偿？？？？   如何实现？？？ 是否必要？？？
    flow = tvl1.calc(onset_gray, apex_gray, None)
    nose_roi = get_nose_region(onset_frame)
    mean_nose_flow = np.mean(flow[nose_roi], axis=(0, 1))
    # 全局补偿：减去鼻尖平均位移
    compensated_flow = flow - mean_nose_flow
    return compensated_flow
    

def flow_to_color_map(flow, onset_gray):
    """光流向色彩空间映射（MMC-Mapping核心）"""
    # 计算运动幅度和角度
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 方向→色相映射  弧度转角度，并进行压缩 （0-360° → 0-180，适配OpenCV HSV）
    """    
    色相意义:
        不同的角度值对应HSV色彩空间中的不同颜色(详细可参考HSV颜色模型)：
          0° = 红色   60° = 黄色   120° = 绿色
          180° = 青色  240° = 蓝色   300° = 品红
    """
    hue = angle * 180 / np.pi / 2
    # 幅度→明度映射（归一化到0-255）
    value = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # 构建HSV图像（饱和度固定最大值）
    """
        H(色相) - 表示运动方向
        S(饱和度) - 固定255（最大值），使颜色饱和鲜艳
        V(明度) - 表示运动幅度
    """
    saturation = 255 * np.ones_like(hue)
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.uint8)
    # 转换为BGR色彩空间
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 叠加到起始帧灰度图   这个权重比例怎样设置？？ 暂时7:3吧
    onset_bgr = cv2.cvtColor(onset_gray, cv2.COLOR_GRAY2BGR)
    mmc_map = cv2.addWeighted(onset_bgr, 0.5, flow_color, 0.5, 0)
    return mmc_map

def flow_to_color_map_onlyColor(flow):
    """光流向色彩空间映射（MMC-Mapping核心）"""
    # 计算运动幅度和角度
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 方向→色相映射（0-360° → 0-180，适配OpenCV HSV）
    hue = angle * 180 / np.pi / 2
    # 幅度→明度映射（归一化到0-255）
    value = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # 构建HSV图像（饱和度固定最大值）
    saturation = 255 * np.ones_like(hue)
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.uint8)
    # 转换为BGR色彩空间
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_color




def generate_mmc_for_all_sequences(parent_folder_path):
    # 获取父文件夹下的所有子文件夹
    subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]
    
    if not subfolders:
        print(f"父文件夹 '{parent_folder_path}' 中没有子文件夹")
        return

    # 遍历每个子文件夹
    for folder_path in subfolders:
        try:
            output_path = os.path.join(folder_path, "mmc_map2_color_no_nose.png")
            print(f"\n处理序列: {os.path.basename(folder_path)}")
            print(f"输出路径: {output_path}")
            
            # 从当前子文件夹加载图像序列
            frames, filenames = load_image_sequence(folder_path)
            if len(frames) < 2:
                print(f"警告: {folder_path} 中图像不足2张，已跳过")
                continue

            # 找到峰值帧索引
            apex_idx = find_apex_frame_index(frames)
            print(f"检测到峰值帧: {filenames[apex_idx]} (索引 {apex_idx})")

            # 计算光流并生成MMC-Map
            flow = compute_optical_flow(frames[0], frames[apex_idx])
            onset_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            mmc_map = flow_to_color_map(flow, onset_gray)

            # mmc_map = flow_to_color_map_onlyColor(flow)

            # 保存结果到当前子文件夹
            cv2.imwrite(output_path, mmc_map)
            print(f"MMC-Map 已保存至: {output_path}")
            
        except Exception as e:
            print(f"处理 {folder_path} 时出错: {str(e)}")



def generate_multiple_flow_maps(folder_path):
    """生成多种光流图：相对于起始帧的光流和相邻帧的光流"""
    # 创建保存结果的子文件夹
    relative_dir = os.path.join(folder_path, "nose_relative_to_onset")
    adjacent_dir = os.path.join(folder_path, "nose_adjacent_frames")
    os.makedirs(relative_dir, exist_ok=True)
    os.makedirs(adjacent_dir, exist_ok=True)
    
    # 加载图像序列
    frames, filenames = load_image_sequence(folder_path)
    if len(frames) < 2:
        print(f"警告: {folder_path} 中图像不足2张，已跳过")
        return
    
    # 找到峰值帧索引
    apex_idx = find_apex_frame_index(frames)
    print(f"检测到峰值帧: {filenames[apex_idx]} (索引 {apex_idx})")
    
    # 确定采样范围（峰值帧前后各5帧）
    start_idx = max(0, apex_idx - 5)
    end_idx = min(len(frames) - 1, apex_idx + 5)
    
    # 获取起始帧灰度图（用于光流叠加）
    onset_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    print(f"采样范围: {start_idx}-{end_idx} (共{end_idx - start_idx + 1}帧)")
    
    # 1. 生成相对于起始帧的光流图（绝对位移）
    print("\n生成相对起始帧的光流图...")
    for i in range(start_idx, end_idx + 1):
        if i == 0:
            continue  # 跳过起始帧自身
            
        # 计算当前帧相对于起始帧的光流
        flow = compute_optical_flow(frames[0], frames[i])
        
        # 生成彩色光流图
        flow_color = flow_to_color_map(flow, onset_gray)
        
        # 保存结果
        output_name = f"relative_to_onset_{filenames[i].split('.')[0]}.png"
        cv2.imwrite(os.path.join(relative_dir, output_name), flow_color)
        print(f"已保存: {output_name}")
    
    # 2. 生成相邻帧之间的光流图（局部运动）
    print("\n生成相邻帧的光流图...")
    for i in range(start_idx + 1, end_idx + 1):
        prev_frame = frames[i-1]
        curr_frame = frames[i]
        
        # 计算前一帧与当前帧之间的光流
        flow = compute_optical_flow(prev_frame, curr_frame)
        
        # 获取前一帧的灰度图用于叠加
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # 生成彩色光流图
        flow_color = flow_to_color_map(flow, prev_gray)
        
        # 保存结果
        output_name = f"adjacent_{filenames[i-1].split('.')[0]}_to_{filenames[i].split('.')[0]}.png"
        cv2.imwrite(os.path.join(adjacent_dir, output_name), flow_color)
        print(f"已保存: {output_name}")


def generate_mul_flow_maps_for_all_sequences(parent_folder_path):
    """处理父文件夹下的所有子文件夹序列"""
    subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]
    
    if not subfolders:
        print(f"父文件夹 '{parent_folder_path}' 中没有子文件夹")
        return

    for folder_path in subfolders:
        try:
            print(f"\n处理序列: {os.path.basename(folder_path)}")
            generate_multiple_flow_maps(folder_path)
        except Exception as e:
            print(f"处理 {folder_path} 时出错: {str(e)}")

if __name__ == "__main__":
    parent_folder = "/home/d1/zwb/yhf/MEGC_unseen_testset/ME_VQA_MEGC_2025_Test_Cropped"
    generate_mul_flow_maps_for_all_sequences(parent_folder)


# if __name__ == "__main__":
#     parent_folder = "/home/d1/zwb/yhf/MEGC_unseen_testset/ME_VQA_MEGC_2025_Test_Cropped"
#     generate_mmc_for_all_sequences(parent_folder)


