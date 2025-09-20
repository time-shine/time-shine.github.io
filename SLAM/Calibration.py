import cv2
import numpy as np
import os
from typing import List, Tuple

# ====================== 1. 核心参数配置（必须根据实际硬件修改）======================
# 棋盘格参数（内角点数量：宽7个，高9个；单个方块实际尺寸，单位：mm）
CHESSBOARD = (7, 9)          # 棋盘格内角点数量（宽×高）
SQUARE_SIZE = 25.0           # 棋盘格单个方块物理尺寸（需实际测量，默认25mm）

# 相机分辨率（输入图像总分辨率3840×1080，左右相机各占一半）
TOTAL_WIDTH = 3840           # 总图像宽度
TOTAL_HEIGHT = 1080          # 总图像高度
LEFT_WIDTH = TOTAL_WIDTH // 2 # 左相机图像宽度（1920）
RIGHT_WIDTH = TOTAL_WIDTH // 2# 右相机图像宽度（1920）

# 标定流程参数
MIN_CALIB_IMGS = 15          # 最少需要的标定图像数量（建议15-20张，保证姿态多样性）
SAVE_DIR = "calib_imgs"      # 存储标定原始图像的文件夹
ESC_KEY = 27                 # 退出采集的按键（ESC）
SAVE_KEY = 32                # 保存图像的按键（空格）

# 角点检测与亚像素优化的终止条件
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ====================== 2. 初始化标定数据容器======================
def init_calib_data() -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    初始化3D世界坐标和2D图像坐标容器
    返回：
        objp: 单张棋盘格的3D坐标模板
        obj_points: 所有图像的3D坐标列表
        img_points_left: 左相机所有图像的2D角点列表
        img_points_right: 右相机所有图像的2D角点列表
    """
    # 生成棋盘格3D坐标（z轴为0，xy轴按棋盘格排列，缩放为实际物理尺寸）
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # 转换为实际物理尺寸（mm）

    # 存储所有标定图像的3D点和2D点
    obj_points = []       # 3D世界坐标（所有图像共用同一套3D点）
    img_points_left = []  # 左相机2D角点坐标
    img_points_right = [] # 右相机2D角点坐标

    return objp, obj_points, img_points_left, img_points_right


# ====================== 3. 创建文件夹（存储标定图像）======================
def create_save_dir(save_dir: str) -> None:
    """创建标定图像保存文件夹，避免路径不存在报错"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"✅ 已创建标定图像保存文件夹：{os.path.abspath(save_dir)}")
    else:
        print(f"ℹ️  标定图像保存文件夹已存在：{os.path.abspath(save_dir)}")


# ====================== 4. 采集标定图像（实时读取相机数据）======================
def capture_calib_imgs(
    objp: np.ndarray,
    save_dir: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    从双目相机采集标定图像，分割左/右相机图像并检测角点
    参数：
        objp: 单张棋盘格的3D坐标模板
        save_dir: 标定图像保存路径
    返回：
        obj_points: 所有图像的3D坐标列表
        img_points_left: 左相机2D角点列表
        img_points_right: 右相机2D角点列表
    """
    # 打开相机（0为默认相机，多相机时需修改索引，如1、2）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ 无法打开相机！请检查相机连接或更换索引（如cv2.VideoCapture(1)）")

    # 设置相机分辨率（必须与硬件输出一致，否则图像会拉伸/压缩）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TOTAL_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TOTAL_HEIGHT)

    # 验证实际分辨率（避免相机不支持设置的分辨率）
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if actual_width != TOTAL_WIDTH or actual_height != TOTAL_HEIGHT:
        print(f"⚠️  相机不支持3840×1080分辨率，实际分辨率：{actual_width:.0f}×{actual_height:.0f}")
        print("⚠️  请确认相机硬件参数，或修改代码中的TOTAL_WIDTH/TOTAL_HEIGHT")

    # 初始化数据容器
    obj_points = []
    img_points_left = []
    img_points_right = []
    img_count = 0

    # 采集提示信息
    print("\n" + "="*50)
    print("📷 开始采集标定图像（左相机：绿色角点 | 右相机：红色角点）")
    print(f"💡 操作说明：")
    print(f"   - 移动棋盘格，覆盖相机所有视场（建议15-20张不同姿态）")
    print(f"   - 按【空格键】保存图像（仅左右相机均检测到角点时有效）")
    print(f"   - 按【ESC键】退出采集（当前已保存：{img_count}/{MIN_CALIB_IMGS}）")
    print("="*50 + "\n")

    while True:
        # 读取相机帧
        ret, frame = cap.read()
        if not ret:
            print("⚠️  无法读取相机帧，跳过...")
            continue

        # 分割左、右相机图像（左半屏：左相机；右半屏：右相机）
        left_frame = frame[:, :LEFT_WIDTH].copy()   # 左相机：列0~1919
        right_frame = frame[:, RIGHT_WIDTH:].copy() # 右相机：列1920~3839

        # 复制图像用于显示（避免修改原始帧）
        left_show = left_frame.copy()
        right_show = right_frame.copy()

        # 转换为灰度图（角点检测需灰度图像）
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格内角点（ret：是否检测成功；corners：角点坐标）
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD, None)

        # 若左右相机均检测到角点，执行亚像素优化并绘制角点
        if ret_left and ret_right:
            # 亚像素级角点优化（提高检测精度，关键步骤）
            corners_left_sub = cv2.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), TERM_CRITERIA
            )
            corners_right_sub = cv2.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), TERM_CRITERIA
            )

            # 绘制角点（左相机绿色，右相机红色）
            cv2.drawChessboardCorners(left_show, CHESSBOARD, corners_left_sub, ret_left)
            cv2.drawChessboardCorners(right_show, CHESSBOARD, corners_right_sub, ret_right)

            # 在图像上添加"可保存"提示
            cv2.putText(left_show, "Press SPACE to Save", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(right_show, "Press SPACE to Save", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 合并左右图像用于显示（保持3840×1080分辨率）
        combined_show = np.hstack((left_show, right_show))
        # 显示已保存图像数量
        cv2.putText(combined_show, f"Saved: {img_count}/{MIN_CALIB_IMGS}",
                    (TOTAL_WIDTH//2 - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # 显示图像窗口（支持缩放）
        cv2.namedWindow("Calibration Window (Left: Green, Right: Red)", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Window (Left: Green, Right: Red)", combined_show)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ESC_KEY:
            # 退出采集时检查图像数量
            if img_count < MIN_CALIB_IMGS:
                print(f"⚠️  警告：已保存{img_count}张图像，少于最少{MIN_CALIB_IMGS}张，标定结果可能不准确！")
            break
        elif key == SAVE_KEY and ret_left and ret_right:
            # 保存原始图像（左、右相机分开保存，便于后续复现）
            left_save_path = os.path.join(save_dir, f"left_{img_count:02d}.png")
            right_save_path = os.path.join(save_dir, f"right_{img_count:02d}.png")
            cv2.imwrite(left_save_path, left_frame)
            cv2.imwrite(right_save_path, right_frame)

            # 保存角点数据（3D点+2D点）
            obj_points.append(objp)
            img_points_left.append(corners_left_sub)
            img_points_right.append(corners_right_sub)

            img_count += 1
            print(f"✅ 已保存第{img_count}张图像 | 左相机：{os.path.basename(left_save_path)}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return obj_points, img_points_left, img_points_right


# ====================== 5. 执行双目相机内参标定======================
def calibrate_stereo(
    obj_points: List[np.ndarray],
    img_points_left: List[np.ndarray],
    img_points_right: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    执行双目相机内参标定（先单目标定，再验证精度）
    参数：
        obj_points: 所有图像的3D坐标列表
        img_points_left: 左相机2D角点列表
        img_points_right: 右相机2D角点列表
    返回：
        mtx_left: 左相机内参矩阵
        dist_left: 左相机畸变系数
        mtx_right: 右相机内参矩阵
        dist_right: 右相机畸变系数
        reproj_err_left: 左相机重投影误差
        reproj_err_right: 右相机重投影误差
    """
    # 左相机单目标定
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        obj_points, img_points_left, (LEFT_WIDTH, TOTAL_HEIGHT), None, None
    )

    # 右相机单目标定
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        obj_points, img_points_right, (RIGHT_WIDTH, TOTAL_HEIGHT), None, None
    )

    # 计算重投影误差（评估标定精度，越小越好，一般<1.0为优秀）
    def calc_reproj_error(
        obj_pts: List[np.ndarray],
        img_pts: List[np.ndarray],
        mtx: np.ndarray,
        dist: np.ndarray,
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray]
    ) -> float:
        total_err = 0.0
        for i in range(len(obj_pts)):
            # 投影3D点到2D图像
            img_pts_reproj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
            # 计算L2误差
            err = cv2.norm(img_pts[i], img_pts_reproj, cv2.NORM_L2) / len(img_pts_reproj)
            total_err += err
        # 平均误差
        return total_err / len(obj_pts)

    # 计算左右相机的重投影误差
    reproj_err_left = calc_reproj_error(obj_points, img_points_left, mtx_left, dist_left, rvecs_left, tvecs_left)
    reproj_err_right = calc_reproj_error(obj_points, img_points_right, mtx_right, dist_right, rvecs_right, tvecs_right)

    # 打印精度报告
    print("\n" + "="*50)
    print("📊 标定精度报告")
    print("="*50)
    print(f"左相机平均重投影误差：{reproj_err_left:.4f} px（越小越好，<1.0为优秀）")
    print(f"右相机平均重投影误差：{reproj_err_right:.4f} px（越小越好，<1.0为优秀）")
    print("="*50 + "\n")

    return mtx_left, dist_left, mtx_right, dist_right, reproj_err_left, reproj_err_right


# ====================== 6. 将标定结果写入a.txt======================
def save_calib_result(
    mtx_left: np.ndarray,
    dist_left: np.ndarray,
    mtx_right: np.ndarray,
    dist_right: np.ndarray,
    reproj_err_left: float,
    reproj_err_right: float,
    obj_points: List[np.ndarray]
) -> None:
    """
    将内参标定结果写入a.txt，包含内参矩阵、畸变系数、标定参数等
    参数：
        mtx_left/right: 左右相机内参矩阵
        dist_left/right: 左右相机畸变系数
        reproj_err_left/right: 左右相机重投影误差
        obj_points: 标定图像数量（通过长度获取）
    """
    # 格式化畸变系数（展平为1D数组）
    dist_left_flat = dist_left.flatten()
    dist_right_flat = dist_right.flatten()

    # 写入文件
    with open("a.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("双目相机内参标定结果\n")
        f.write("="*60 + "\n")
        f.write("【标定基础参数】\n")
        f.write(f"  棋盘格内角点数量：{CHESSBOARD[0]} × {CHESSBOARD[1]}\n")
        f.write(f"  棋盘格方块尺寸：{SQUARE_SIZE} mm\n")
        f.write(f"  相机分辨率：{TOTAL_WIDTH}×{TOTAL_HEIGHT}（左：{LEFT_WIDTH}×{TOTAL_HEIGHT}，右：{RIGHT_WIDTH}×{TOTAL_HEIGHT}）\n")
        f.write(f"  标定图像数量：{len(obj_points)} 张\n")
        f.write(f"  标定时间：{os.popen('date +%Y-%m-%d_%H:%M:%S').read().strip()}（系统时间）\n")
        f.write("\n")

        f.write("="*60 + "\n")
        f.write("【左相机内参矩阵（3×3，单位：像素）】\n")
        f.write("="*60 + "\n")
        f.write(f"  焦距 fx: {mtx_left[0, 0]:.6f}\n")
        f.write(f"  焦距 fy: {mtx_left[1, 1]:.6f}\n")
        f.write(f"  主点 cx: {mtx_left[0, 2]:.6f}（x轴主点坐标）\n")
        f.write(f"  主点 cy: {mtx_left[1, 2]:.6f}（y轴主点坐标）\n")
        f.write("  完整矩阵：\n")
        for row in mtx_left:
            f.write(f"    {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}\n")
        f.write("\n")

        f.write("="*60 + "\n")
        f.write("【左相机畸变系数（5参数模型：k1, k2, p1, p2, k3）】\n")
        f.write("="*60 + "\n")
        f.write(f"  k1: {dist_left_flat[0]:.8f}（径向畸变系数1）\n")
        f.write(f"  k2: {dist_left_flat[1]:.8f}（径向畸变系数2）\n")
        f.write(f"  p1: {dist_left_flat[2]:.8f}（切向畸变系数1）\n")
        f.write(f"  p2: {dist_left_flat[3]:.8f}（切向畸变系数2）\n")
        f.write(f"  k3: {dist_left_flat[4]:.8f}（径向畸变系数3）\n")
        f.write(f"  平均重投影误差：{reproj_err_left:.4f} px\n")
        f.write("\n")

        f.write("="*60 + "\n")
        f.write("【右相机内参矩阵（3×3，单位：像素）】\n")
        f.write("="*60 + "\n")
        f.write(f"  焦距 fx: {mtx_right[0, 0]:.6f}\n")
        f.write(f"  焦距 fy: {mtx_right[1, 1]:.6f}\n")
        f.write(f"  主点 cx: {mtx_right[0, 2]:.6f}（x轴主点坐标）\n")
        f.write(f"  主点 cy: {mtx_right[1, 2]:.6f}（y轴主点坐标）\n")
        f.write("  完整矩阵：\n")
        for row in mtx_right:
            f.write(f"    {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}\n")
        f.write("\n")

        f.write("="*60 + "\n")
        f.write("【右相机畸变系数（5参数模型：k1, k2, p1, p2, k3）】\n")
        f.write("="*60 + "\n")
        f.write(f"  k1: {dist_right_flat[0]:.8f}（径向畸变系数1）\n")
        f.write(f"  k2: {dist_right_flat[1]:.8f}（径向畸变系数2）\n")
        f.write(f"  p1: {dist_right_flat[2]:.8f}（切向畸变系数1）\n")
        f.write(f"  p2: {dist_right_flat[3]:.8f}（切向畸变系数2）\n")
        f.write(f"  k3: {dist_right_flat[4]:.8f}（径向畸变系数3）\n")
        f.write(f"  平均重投影误差：{reproj_err_right:.4f} px\n")
        f.write("\n")

        f.write("="*60 + "\n")
        f.write("【使用说明】\n")
        f.write("="*60 + "\n")
        f.write("1. 内参矩阵（K）用于将3D世界坐标转换为2D图像坐标\n")
        f.write("2. 畸变系数用于矫正相机畸变（调用cv2.undistort()）\n")
        f.write("3. 重投影误差<1.0为优秀，<2.0为可接受，>2.0建议重新标定\n")
        f.write(f"4. 文件路径：{os.path.abspath('a.txt')}\n")
        f.write("="*60 + "\n")

    print(f"✅ 标定结果已保存到：{os.path.abspath('a.txt')}")


# ====================== 7. 主函数（串联全流程）======================
def main():
    print("="*60)
    print("        双目相机内参标定程序")
    print(f"        分辨率：{TOTAL_WIDTH}×{TOTAL_HEIGHT} | 棋盘格：{CHESSBOARD[0]}×{CHESSBOARD[1]}")
    print("="*60)

    try:
        # 步骤1：初始化数据和文件夹
        objp, obj_points, img_points_left, img_points_right = init_calib_data()
        create_save_dir(SAVE_DIR)

        # 步骤2：采集标定图像
        print("\n【步骤1/3】采集标定图像...")
        obj_points, img_points_left, img_points_right = capture_calib_imgs(objp, SAVE_DIR)

        # 检查图像数量是否足够
        if len(obj_points) < MIN_CALIB_IMGS:
            raise ValueError(f"❌ 标定图像数量不足（仅{len(obj_points)}张，需至少{MIN_CALIB_IMGS}张），终止标定！")

        # 步骤3：执行内参标定
        print("\n【步骤2/3】执行内参标定...")
        mtx_left, dist_left, mtx_right, dist_right, reproj_err_left, reproj_err_right = calibrate_stereo(
            obj_points, img_points_left, img_points_right
        )

        # 步骤4：保存标定结果
        print("\n【步骤3/3】保存标定结果...")
        save_calib_result(
            mtx_left, dist_left, mtx_right, dist_right,
            reproj_err_left, reproj_err_right, obj_points
        )

        print("\n🎉 标定流程全部完成！")

    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        exit(1)


if __name__ == "__main__":
    main()