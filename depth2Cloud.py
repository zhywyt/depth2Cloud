import os
import numpy as np
import cv2
from path import Path
from tqdm import tqdm


def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()


def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) * scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

    return points

def ReadRT(poses_file):
    data = np.loadtxt(poses_file, dtype=float, delimiter=' ')
    # data like 3x4
    # return 4x4
    # return np.vstack((data, np.array([0, 0, 0, 1])))
    data = np.linalg.inv(np.vstack((data, np.array([0, 0, 0, 1]))))
    # print(data)
    return data
def ReadCam(filename):
    data = np.zeros((3,3),dtype=np.float64)
    with open(filename) as file:
        lines = file.readlines()
        row = 0
        for line in lines:
            if row == 0:
                row+=1
                continue
            elif  row == 4:
                break
            list = line.strip('\n').split(' ')
            data[row-1:] = list[0:3]
            row+=1
    print(data)
    return  data


# image_files: XXXXXX.png (RGB, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# poses: camera-to-world, 4×4 matrix in homogeneous coordinates
def build_point_cloud(dataset_path, scale):
    K = ReadCam(os.path.join(dataset_path , "out/readme.txt"))
    image_files = sorted(Path(os.path.join(dataset_path,"out/rgbd")).files('rgb_*.jpg'))
    depth_files = sorted(Path(os.path.join(dataset_path, "out/rgbd")).files('depth_*.png'))
    poses_files = sorted(Path(os.path.join(dataset_path, "out/pose_1")).files('RT_*.txt'))

    current_points_3D = []
    # 间隔设置为5
    # for i in tqdm([0,1]):
    for i in tqdm(range(0, len(image_files),5)):
        image_file = image_files[i]
        depth_file = depth_files[i]

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)

        current_points_3D += depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=ReadRT(poses_files[i]))
    save_ply_name = os.path.basename(os.path.splitext(image_files[i])[0]) + ".ply"
    save_ply_path = os.path.join(dataset_path, "point_clouds")
    # 计算模型的AABB包围盒的两个顶点

    min_vert = np.min(current_points_3D, axis=0)[:3]
    max_vert = np.max(current_points_3D, axis=0)[:3]
    print("min: ", min_vert)
    print("max: ", max_vert)

    if not os.path.exists(save_ply_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.mkdir(save_ply_path)
    # write_point_cloud(os.path.join(save_ply_path, save_ply_name), current_points_3D)


if __name__ == '__main__':
    index = "002"
    dataset_folder = os.path.join("../../../fusai")
    scene = index
    scale_factor = 3000.0/65535.0
    for i in range(1,7):
        scene = "00" + str(i)
        build_point_cloud(os.path.join(dataset_folder, scene), scale_factor)
