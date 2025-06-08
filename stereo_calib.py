import numpy as np
import cv2
import glob
import pickle
import os

def print_formatted_pickle(file_path):
    """格式化打印pickle文件内容"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("="*50 + "\n标定结果详细内容:\n" + "="*50)
    
    for key in sorted(data.keys()):
        val = data[key]
        
        print(f"\n{key} ({type(val).__name__}):")
        if isinstance(val, np.ndarray):
            print(np.array2string(val, precision=4, suppress_small=True))
        else:
            print(val)

# 棋盘格的内部角点数（格子数-1）
chessboard_size = (7, 10)  # 假设使用8x6的棋盘格
chessboard_square_size = 15.0  # 棋盘格每个方格的边长，单位为毫米
# 定义世界坐标系中的角点位置 (Z=0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= chessboard_square_size  # 乘上实际尺寸（重要！）
# 存储在图像中检测到的角点
objpoints = []  # 3D点在世界坐标系中
imgpoints_left = []  # 左相机2D图像点
imgpoints_right = []  # 右相机2D图像点


# 左右相机图像路径
left_images = sorted(glob.glob('photo/left/*.jpg'))
right_images = sorted(glob.glob('photo/right/*.jpg'))
print(left_images)
print(right_images)
# 检测角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for left_img, right_img in zip(left_images, right_images):
    # 读取图像
    img_l = cv2.imread(left_img)
    img_r = cv2.imread(right_img)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    # 查找角点
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)
    print(ret_l, ret_r)
    # 如果两个图像都检测到角点
    if ret_l and ret_r:
        objpoints.append(objp)
        
        # 亚像素精确化
        corners_l_sub = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        corners_r_sub = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
        
        imgpoints_left.append(corners_l_sub)
        imgpoints_right.append(corners_r_sub)
        
        # 可视化（可选）
        cv2.drawChessboardCorners(img_l, chessboard_size, corners_l_sub, ret_l)
        cv2.drawChessboardCorners(img_r, chessboard_size, corners_r_sub, ret_r)
        cv2.imshow('left', img_l)
        cv2.imshow('right', img_r)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 左相机标定
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_l.shape[::-1], None, None)

# 右相机标定
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_r.shape[::-1], None, None)

# 立体标定
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC  # 固定内参，只计算外参

ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1],
    criteria=criteria, flags=flags)

print("Rotation matrix:\n", R)
print("Translation vector:\n", T)

# 立体校正
rectify_scale = 0  # 0=完全校正会裁剪图像，1=不裁剪图像
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, 
    gray_l.shape[::-1], R, T,
    alpha=rectify_scale)

print("Q matrix:\n", Q)

# 计算校正映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, gray_l.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, gray_r.shape[::-1], cv2.CV_16SC2)

# 保存标定参数
calib_result = {
    'cameraMatrix_left': mtx_l,
    'distCoeffs_left': dist_l,
    'cameraMatrix_right': mtx_r,
    'distCoeffs_right': dist_r,
    'R': R,
    'T': T,
    'E': E,
    'F': F,
    'R1': R1,
    'R2': R2,
    'P1': P1,
    'P2': P2,
    'Q': Q,
    'roi1': roi1,
    'roi2': roi2,
    'left_map1': left_map1,
    'left_map2': left_map2,
    'right_map1': right_map1,
    'right_map2': right_map2
}

with open('stereo_calibration.pkl', 'wb') as f:
    pickle.dump(calib_result, f)

# 保存主标定参数为XML
fs = cv2.FileStorage('stereo_calibration.xml', cv2.FILE_STORAGE_WRITE)
for key in ['cameraMatrix_left', 'distCoeffs_left',
           'cameraMatrix_right', 'distCoeffs_right',
           'R', 'T', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q']:
    fs.write(key, calib_result[key])
fs.release()

# 单独保存remap映射为XML
fs_remap = cv2.FileStorage('stereo_remap.xml', cv2.FILE_STORAGE_WRITE)
fs_remap.write('map11', calib_result['left_map1'])
fs_remap.write('map12', calib_result['left_map2'])
fs_remap.write('map21', calib_result['right_map1'])
fs_remap.write('map22', calib_result['right_map2'])
fs_remap.release()

print_formatted_pickle('stereo_calibration.pkl')
# 加载测试图像
left_test = cv2.imread('photo/left/left0.jpg')
right_test = cv2.imread('photo/right/right0.jpg')

# 校正图像
left_rectified = cv2.remap(left_test, left_map1, left_map2, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_test, right_map1, right_map2, cv2.INTER_LINEAR)

# 绘制水平线（用于验证校正效果）
for y in range(0, left_rectified.shape[0], 50):
    cv2.line(left_rectified, (0, y), (left_rectified.shape[1], y), (0, 255, 0), 1)
    cv2.line(right_rectified, (0, y), (right_rectified.shape[1], y), (0, 255, 0), 1)

# 并排显示结果
result = np.hstack((left_rectified, right_rectified))
cv2.imshow('Rectified Images', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
