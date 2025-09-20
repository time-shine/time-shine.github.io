# extract_calibration_data.py
import numpy as np
import json

# 加载之前保存的标定数据
data = np.load('camera_calibration.npz')

# 提取相机内参矩阵和畸变系数
mtx = data['mtx'].tolist()   # 转换为列表
dist = data['dist'].tolist() # 转换为列表

# 准备一个字典来存储数据
calibration_data = {
    "camera_matrix": mtx,
    "distortion_coefficients": dist
}

# 将数据保存为 JSON 文件
with open('camera_calibration.json', 'w') as f:
    json.dump(calibration_data, f, indent=4)

print("相机内参矩阵:")
print(mtx)
print("\n畸变系数:")
print(dist)
print("\n数据已保存到 camera_calibration.json")