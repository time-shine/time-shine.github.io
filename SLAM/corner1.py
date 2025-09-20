import cv2
import numpy as np
import time

def optimized_corner_detection():
    # 打开默认摄像头（摄像头0）
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 优化角点检测参数（针对1280x720分辨率）
    feature_params = dict(
        maxCorners=200,           # 增加最大角点数量以适应更高分辨率
        qualityLevel=0.01,        # 降低质量水平以检测更多角点
        minDistance=10,           # 增加最小距离以避免角点过于密集
        blockSize=9,              # 增加块大小以适应更高分辨率
        useHarrisDetector=False,  # 使用Shi-Tomasi方法
        k=0.04                    # Harris角点检测器的自由参数
    )
    
    # 创建CLAHE对象（限制对比度自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    print("开始实时角点检测，按'q'键退出")
    
    # 性能监控变量
    prev_time = time.time()
    fps = 0
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用CLAHE增强对比度，有助于角点检测
        gray = clahe.apply(gray)
        
        # 使用高斯模糊减少噪声
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Shi-Tomasi方法检测角点
        corners = cv2.goodFeaturesToTrack(gray_blur, mask=None, **feature_params)
        
        # 绘制角点
        if corners is not None:
            corners = np.int8(corners)
            for i, corner in enumerate(corners):
                x, y = corner.ravel()
                # 根据角点质量绘制不同大小的圆
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                # 在强角点处绘制外圈
                if i % 5 == 0:  # 每5个角点选一个作为强角点
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), 1)
        
        # 计算并显示FPS
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
        prev_time = current_time
        
        # 在图像上显示信息
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Corners: {len(corners) if corners is not None else 0}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Resolution: 1280x720", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('Optimized Corner Detection', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    optimized_corner_detection()