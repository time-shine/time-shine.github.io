import cv2
import numpy as np
import os
from typing import List, Tuple

# ====================== 1. 核心参数配置（关键：降低角度相关精度要求）======================
# 棋盘格参数（必须与实际使用的棋盘格匹配！）
CHESSBOARD = (9, 6)          # 棋盘格内角点数量（宽×高，不变）
SQUARE_SIZE = 10.0           # 棋盘格单个方块物理尺寸（单位：mm，不变）

# 单目相机分辨率（用户指定1280×720，不变）
CAM_WIDTH = 1280             
CAM_HEIGHT = 720             

# 标定流程参数（可选：降低最少图像数量，配合角度要求放宽）
MIN_CALIB_IMGS = 10          # 【修改】从15张降至10张，减少对多姿态的依赖
SAVE_DIR = "mono_calib_imgs" 
ESC_KEY = 27                 
SAVE_KEY = 32                

# 角点检测与亚像素优化终止条件（【核心修改】放宽精度要求）
# 原：(30次迭代, 0.001精度) → 新：(10次迭代, 0.01精度)，降低亚像素收敛门槛
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)


# ====================== 2. 初始化标定数据容器（无修改）======================
def init_calib_data() -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  

    obj_points = []  
    img_points = []  

    return objp, obj_points, img_points


# ====================== 3. 创建标定图像保存文件夹（无修改）======================
def create_save_dir(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"✅ 已创建单目标定图像文件夹：{os.path.abspath(save_dir)}")
    else:
        print(f"ℹ️  单目标定图像文件夹已存在：{os.path.abspath(save_dir)}")


# ====================== 4. 采集单目标定图像（核心修改：放宽角点检测与预处理）======================
def capture_calib_imgs(
    objp: np.ndarray,
    save_dir: str
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ 无法打开相机！请检查：1.相机连接 2.修改索引（如cv2.VideoCapture(0)）")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if actual_width != CAM_WIDTH or actual_height != CAM_HEIGHT:
        print(f"⚠️  警告：相机不支持1280×720分辨率！实际分辨率：{actual_width:.0f}×{actual_height:.0f}")
        print("⚠️  请修改代码中 CAM_WIDTH/CAM_HEIGHT 为相机支持的分辨率（如640×480）")

    obj_points = []
    img_points = []
    img_count = 0

    print("\n" + "="*50)
    print("📷 单目相机标定图像采集（已降低角度质量要求）")
    print(f"💡 操作说明：")
    print(f"   - 移动棋盘格，允许更大角度偏差（最少{MIN_CALIB_IMGS}张即可）")
    print(f"   - 绿色角点显示时，按【空格键】保存图像")
    print(f"   - 按【ESC键】退出采集（当前已保存：{img_count}/{MIN_CALIB_IMGS}）")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  无法读取相机帧，跳过...")
            continue

        frame_show = frame.copy()
        # 1. 图像预处理（【修改】简化处理，避免过度增强丢失角点）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 注释掉对比度增强和降噪（过度处理会让角度偏差大的棋盘格角点更难检测）
        # gray = cv2.equalizeHist(gray)  
        # gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=1)  

        # 2. 角点检测（【核心修改】移除严格搜索flag，放宽角度容忍度）
        # 原flag：ADAPTIVE_THRESH + EXHAUSTIVE + FAST_CHECK + NORMALIZE_IMAGE
        # 新flag：仅保留基础阈值和快速检测，移除EXHAUSTIVE（严格搜索）和NORMALIZE_IMAGE（归一化）
        ret_corner, corners = cv2.findChessboardCorners(
            gray, 
            CHESSBOARD, 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +  # 保留：自适应阈值（基础功能）
                  cv2.CALIB_CB_FAST_CHECK        # 保留：快速检测（减少严格校验）
        )

        # 若角点检测成功，执行亚像素优化（【修改】缩小亚像素窗口，降低精度要求）
        if ret_corner:
            # 原winSize=(11,11) → 新winSize=(5,5)：窗口越小，对角度偏差的敏感度越低
            corners_sub = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), TERM_CRITERIA
            )
            cv2.drawChessboardCorners(frame_show, CHESSBOARD, corners_sub, ret_corner)
            cv2.putText(frame_show, "Press SPACE to Save", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame_show, f"Saved: {img_count}/{MIN_CALIB_IMGS}",
                    (CAM_WIDTH//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        cv2.namedWindow("Mono Calibration Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Mono Calibration Window", frame_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ESC_KEY:
            if img_count < MIN_CALIB_IMGS:
                print(f"⚠️  警告：已保存{img_count}张图像，少于最少{MIN_CALIB_IMGS}张，标定结果可能不准确！")
            break
        elif key == SAVE_KEY and ret_corner:
            save_path = os.path.join(save_dir, f"mono_{img_count:02d}.png")
            cv2.imwrite(save_path, frame)

            obj_points.append(objp)
            img_points.append(corners_sub)

            img_count += 1
            print(f"✅ 已保存第{img_count}张图像 | 路径：{os.path.basename(save_path)}")

    cap.release()
    cv2.destroyAllWindows()
    return obj_points, img_points


# ====================== 5. 执行单目标定（【修改】放宽重投影误差评价标准）======================
def calibrate_mono(
    obj_points: List[np.ndarray],
    img_points: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, float]:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (CAM_WIDTH, CAM_HEIGHT), None, None
    )

    # 计算平均重投影误差（逻辑不变，评价标准后续修改）
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
            img_pts_reproj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
            err = cv2.norm(img_pts[i], img_pts_reproj, cv2.NORM_L2) / len(img_pts_reproj)
            total_err += err
        return total_err / len(obj_pts)

    reproj_err = calc_reproj_error(obj_points, img_points, mtx, dist, rvecs, tvecs)

    # 【修改】放宽精度评价：原<1优秀/<2可接受 → 新<1.5优秀/<3可接受，匹配角度要求降低
    print("\n" + "="*50)
    print("📊 单目标定精度报告（已放宽评价标准）")
    print("="*50)
    print(f"平均重投影误差：{reproj_err:.4f} px")
    if reproj_err < 1.5:
        eval_str = "优秀"
    elif reproj_err < 3.0:
        eval_str = "可接受"
    else:
        eval_str = "较差（建议重标）"
    print(f"精度评价：{eval_str}")
    print("="*50 + "\n")

    return mtx, dist, reproj_err


# ====================== 6. 保存标定结果到文件（【修改】同步更新误差评价说明）======================
def save_calib_result(
    mtx: np.ndarray,
    dist: np.ndarray,
    reproj_err: float,
    obj_points: List[np.ndarray]
) -> None:
    dist_flat = dist.flatten()
    result_path = "mono_calib_result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("单目相机标定结果（已降低角度质量要求）\n")
        f.write("="*60 + "\n")
        f.write("【标定基础参数】\n")
        f.write(f"  棋盘格内角点数量：{CHESSBOARD[0]} × {CHESSBOARD[1]}\n")
        f.write(f"  棋盘格方块尺寸：{SQUARE_SIZE} mm\n")
        f.write(f"  相机分辨率：{CAM_WIDTH} × {CAM_HEIGHT}（像素）\n")
        f.write(f"  标定图像数量：{len(obj_points)} 张\n")
        f.write(f"  标定时间：{os.popen('date +%Y-%m-%d_%H:%M:%S').read().strip()}（系统时间）\n")
        f.write("\n")

        f.write("="*60 + "\n")
        f.write("【单目相机内参矩阵（3×3，单位：像素）】\n")
        f.write("="*60 + "\n")
        f.write(f"  水平焦距 fx: {mtx[0, 0]:.6f}\n")
        f.write(f"  垂直焦距 fy: {mtx[1, 1]:.6f}\n")
        f.write(f"  主点坐标 cx: {mtx[0, 2]:.6f}（x轴中心）\n")
        f.write(f"  主点坐标 cy: {mtx[1, 2]:.6f}（y轴中心）\n")
        f.write("  完整矩阵：\n")
        for row in mtx:
            f.write(f"    {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}\n")
        f.write("\n")

        f.write("="*60 + "\n")
        f.write("【单目相机畸变系数（5参数模型）】\n")
        f.write("="*60 + "\n")
        f.write(f"  k1: {dist_flat[0]:.8f}（径向畸变系数1）\n")
        f.write(f"  k2: {dist_flat[1]:.8f}（径向畸变系数2）\n")
        f.write(f"  p1: {dist_flat[2]:.8f}（切向畸变系数1）\n")
        f.write(f"  p2: {dist_flat[3]:.8f}（切向畸变系数2）\n")
        f.write(f"  k3: {dist_flat[4]:.8f}（径向畸变系数3）\n")
        f.write(f"  平均重投影误差：{reproj_err:.4f} px\n")
        f.write("  误差评价标准（已放宽）：<1.5px优秀 / <3.0px可接受 / >3.0px建议重标\n")  # 【修改】同步说明
        f.write("\n")

        f.write("="*60 + "\n")
        f.write("【使用说明】\n")
        f.write("="*60 + "\n")
        f.write("1. 畸变矫正：调用 cv2.undistort(frame, mtx, dist, None, newcameramtx) \n")
        f.write("2. 3D→2D投影：使用内参矩阵 mtx 计算图像坐标 \n")
        f.write("3. 本标定已降低角度质量要求，允许棋盘格更大角度偏差\n")  # 【新增】标注修改
        f.write(f"4. 结果文件路径：{os.path.abspath(result_path)}\n")
        f.write("="*60 + "\n")

    print(f"✅ 标定结果已保存到：{os.path.abspath(result_path)}")


# ====================== 7. 主函数（无修改，串联流程）======================
def main():
    print("="*60)
    print("        单目相机标定程序（已降低角度质量要求）")
    print(f"        分辨率：{CAM_WIDTH}×{CAM_HEIGHT} | 棋盘格：{CHESSBOARD[0]}×{CHESSBOARD[1]}")
    print("="*60)

    try:
        objp, obj_points, img_points = init_calib_data()
        create_save_dir(SAVE_DIR)

        print("\n【步骤1/3】采集标定图像...")
        obj_points, img_points = capture_calib_imgs(objp, SAVE_DIR)

        if len(obj_points) < MIN_CALIB_IMGS:
            raise ValueError(f"❌ 标定图像数量不足（仅{len(obj_points)}张，需至少{MIN_CALIB_IMGS}张），终止标定！")

        print("\n【步骤2/3】执行单目标定...")
        mtx, dist, reproj_err = calibrate_mono(obj_points, img_points)

        print("\n【步骤3/3】保存标定结果...")
        save_calib_result(mtx, dist, reproj_err, obj_points)

        print("\n🎉 单目标定流程全部完成！")

    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        exit(1)


if __name__ == "__main__":
    main()