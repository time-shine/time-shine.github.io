import cv2
import numpy as np
import time
import os
import glob

def camera_calibration():
    # 创建标定所需的目录
    if not os.path.exists('calibration_images'):
        os.makedirs('calibration_images')
    
    # 棋盘格参数
    chessboard_size = (8, 5)  # 内部角点数量 (width, height)
    square_size = 10.0  # 棋盘格方格大小（毫米）
    
    # 准备物体点（真实世界中的3D点）
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 存储物体点和图像点的数组
    objpoints = []  # 真实世界中的3D点
    imgpoints = []  # 图像中的2D点
    
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("相机标定程序")
    print("按 's' 键保存当前帧用于标定")
    print("按 'c' 键开始标定计算")
    print("按 'q' 键退出")
    
    image_count = 0
    calibration_done = False
    mtx = None  # 相机内参矩阵
    dist = None  # 畸变系数
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # 如果找到角点，进行亚像素精确化并绘制
        if ret_corners:
            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 绘制角点和棋盘格
            cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, ret_corners)
            
            # 显示状态信息
            cv2.putText(frame, "Chessboard detected - Press 's' to save", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No chessboard detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示已保存的图像数量
        cv2.putText(frame, f"Saved images: {image_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 如果标定已完成，显示标定结果
        if calibration_done:
            cv2.putText(frame, "Calibration completed!", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 应用畸变校正
            if mtx is not None and dist is not None:
                h, w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
                
                # 显示校正后的图像
                cv2.imshow('Undistorted Image', dst)
        
        # 显示原始图像
        cv2.imshow('Camera Calibration', frame)
        
        # 键盘操作
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and ret_corners:
            # 保存图像和角点数据
            img_name = f"calibration_images/calib_{image_count:03d}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"Saved {img_name}")
            
            # 保存角点数据
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            image_count += 1
        elif key == ord('c') and image_count >= 5:  # 至少需要5张图像进行标定
            print("开始标定计算...")
            
            # 进行相机标定
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            
            if ret:
                calibration_done = True
                print("标定成功!")
                
                # 打印标定结果
                print("\n相机内参矩阵:")
                print(mtx)
                print("\n畸变系数:")
                print(dist)
                
                # 计算重投影误差
                mean_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error
                
                print(f"\n平均重投影误差: {mean_error/len(objpoints):.5f} 像素")
                
                # 保存标定结果
                np.savez('camera_calibration.npz', mtx=mtx, dist=dist)
                print("标定结果已保存到 camera_calibration.npz")
            else:
                print("标定失败，请尝试使用更多不同角度的图像")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 如果没有实时标定，可以加载已保存的图像进行标定
    if not calibration_done and len(glob.glob('calibration_images/*.jpg')) >= 5:
        perform_calibration_from_saved_images(chessboard_size, square_size)

def perform_calibration_from_saved_images(chessboard_size, square_size):
    """从已保存的图像进行标定"""
    print("从已保存的图像进行标定...")
    
    # 准备物体点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []  # 真实世界中的3D点
    imgpoints = []  # 图像中的2D点
    
    # 获取所有标定图像
    images = glob.glob('calibration_images/*.jpg')
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            
            # 绘制并显示角点
            cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
            cv2.imshow('Calibration Image', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    if len(objpoints) >= 5:
        # 进行相机标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            print("标定成功!")
            print("\n相机内参矩阵:")
            print(mtx)
            print("\n畸变系数:")
            print(dist)
            
            # 保存标定结果
            np.savez('camera_calibration.npz', mtx=mtx, dist=dist)
            print("标定结果已保存到 camera_calibration.npz")
            
            # 计算重投影误差
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            print(f"\n平均重投影误差: {mean_error/len(objpoints):.5f} 像素")
        else:
            print("标定失败")
    else:
        print("需要至少5张有效的标定图像")

if __name__ == "__main__":
    camera_calibration()