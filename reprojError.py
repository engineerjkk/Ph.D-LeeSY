import cv2
import os
import numpy as np
import collections
from pathlib import Path
from tqdm import tqdm
from hloc.utils.parsers import parse_image_lists, parse_retrieval, names_to_pair

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

VoxelImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec",  "name",  "xys", "point3D_ids", "xyzs", "voxelIDs" ])

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

#쿼터니온을 로테이션매트릭스로 변환
class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

#images.txt 파일 읽어오기.
def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            #print("첫번째 줄 : ", line)
            #print
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                print()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

#cameras.txt 파일 읽어오기
def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            print("첫번째 줄 : ", line)
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

#3D 포인트값있는 txt 파일 읽어오기
def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D
tar_points_strings = []
hloc_images = read_images_text('outputs/aachen/sfm_sift/images.txt')
hloc_points3Ds = read_points3D_text('outputs/aachen/sfm_sift/points3D.txt')
hloc_cameras = read_cameras_text('outputs/aachen/sfm_sift/cameras.txt')

keys = [630, 256, 3154, 3125, 3122, 3118, 2286, 1167, 3099, 1746, 1173, 3095, 276, 2121]
#for key in hloc_images.keys():

########3D->2D를 위한#############
# params2 = hloc_cameras[db_id].params
# camera_metrix2 = np.zeros((3, 3), dtype='float32')
# camera_metrix2[0, 0] = params2[0]  # focal_len
# camera_metrix2[1, 1] = params2[0]
# camera_metrix2[0, 2] = params2[1]
# camera_metrix2[1, 2] = params2[2]
# camera_metrix2[2, 2] = 1
#################################

for i in range(len(keys)):
    key = keys[i]
    tar_imgs = hloc_images[key]
    tar_img = hloc_images[key]
    tar_cameras = hloc_cameras[key]

    tar_camera_header = [tar_cameras.id, tar_cameras.width, tar_cameras.height, *tar_cameras.params]
    tar_image_header = [tar_img.id, *tar_img.qvec, *tar_img.tvec, tar_img.camera_id, tar_img.name]
    print("tar_image_header[9]:",tar_image_header[9])
    tar_points_strings = []
    for xy, point3D_id in zip(tar_img.xys, tar_img.point3D_ids):
        if(point3D_id == -1):
            continue
        tar_points_strings.append(" ".join(map(str, [*xy, point3D_id])))
    print("tar_image_header[9]:",tar_image_header[9])
    image_dir =Path('datasets/aachen/images/images_upright/')
    path = image_dir / tar_image_header[9]
    #path='/home/kangjunekoo/Downloads/Hierarchical-Localization-master/datasets/aachen/images/images_upright/db/3831.jpg'
    image = cv2.imread(str(path))

    tar_points_nums = []
    tar_pt_2ds = []
    tar_pt_3ds = []
    ##################################
    for str_points in tar_points_strings:
        print("\nstr_points :",str_points)
        list_str = str_points.split(" ")
        tmp1 = float(list_str[0])
        tmp2 = float(list_str[1])
        point3D_id = int(list_str[2])
        tar_points_nums.append([tmp1, tmp2, point3D_id])

        tar_pt_2ds.append([tmp1, tmp2])

        tar_3dPt = hloc_points3Ds[point3D_id]
        tar_pt_3ds.append([*tar_3dPt.xyz])
        print("\ntar_pt_3ds ",tar_pt_3ds )
        print("\nint(tmp1)",int(tmp1))
        print("\nint(tmp2)",int(tmp2))
        #이미지에 포인트 위치를 뿌려보자
        image=cv2.circle(image, (int(tmp1), int(tmp2)), 1, (0,255,0), -1)

        #imgpoints2, _ = cv2.projectPoints(tar_pt_3ds, rvec, tvec, camera_metrix, distort)
    cv2.imshow('image', image)
    cv2.waitKey(0)

