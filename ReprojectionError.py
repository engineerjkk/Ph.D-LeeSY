import cv2
import numpy as np
import collections
from pathlib import Path

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

def read_points3D_text(path):

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

hloc_images = read_images_text('outputs/aachen/sfm_sift/images.txt')
hloc_points3Ds = read_points3D_text('outputs/aachen/sfm_sift/points3D.txt')
hloc_cameras = read_cameras_text('outputs/aachen/sfm_sift/cameras.txt')

keys = [130,255,111,630]

for i in range(len(keys)):
    key = keys[i]
    tar_img = hloc_images[key]
    tar_cameras = hloc_cameras[key]
    tar_camera_header = [tar_cameras.id, tar_cameras.width, tar_cameras.height, *tar_cameras.params]
    tar_image_header = [tar_img.id, *tar_img.qvec, *tar_img.tvec, tar_img.camera_id, tar_img.name]
    tar_points_strings = []

    for xy, point3D_id in zip(tar_img.xys, tar_img.point3D_ids):
        if(point3D_id == -1):
            continue
        tar_points_strings.append(" ".join(map(str, [*xy, point3D_id])))

    image_dir =Path('datasets/aachen/images/images_upright/')
    path = image_dir / tar_image_header[9]
    image = cv2.imread(str(path))

    tar_pt_3ds = []
    mean_error=0
    for str_points in tar_points_strings:
        list_str = str_points.split(" ")
        tmp1 = float(list_str[0])
        tmp2 = float(list_str[1])
        point3D_id = int(list_str[2])
        tar_3dPt = hloc_points3Ds[point3D_id]
        tar_pt_3ds.append(tar_3dPt.xyz)
        print("실제 2D x좌표 : ",int(tmp1))
        print("실제 2D y좌표 : ",int(tmp2))

        image=cv2.circle(image, (int(tmp1), int(tmp2)), 1, (0,255,0), -1)
        tar_quat = np.array(tar_image_header[1:5])
        rvec=qvec2rotmat(tar_quat)
        tar_trans = np.array(tar_image_header[5:8])

        camera_metrix = np.zeros((3, 3), dtype='float32')
        camera_metrix[0, 0] = tar_camera_header[3]
        camera_metrix[1, 1] = tar_camera_header[3]
        camera_metrix[0, 2] = tar_camera_header[4]
        camera_metrix[1, 2] = tar_camera_header[5]
        camera_metrix[2, 2] = 1
        distort=tar_camera_header[6]


        imgpoints2, _ = cv2.projectPoints(tar_3dPt.xyz, rvec, tar_trans, camera_metrix, distort)
        imgpoints2 = np.concatenate(imgpoints2).tolist()
        imgpoints2 = np.concatenate(imgpoints2).tolist()

        tmp3 = float(imgpoints2[0])
        tmp4 = float(imgpoints2[1])

        print("3D 좌표 : ", *tar_3dPt.xyz)
        print("재사영된 2D x좌표 : ",int(tmp3))
        print("재사영된 2D y좌표 : ",int(tmp4))

        image = cv2.circle(image, (int(tmp3), int(tmp4)), 1, (0, 0, 255), -1)
        error=cv2.norm((int(tmp1), int(tmp2)),(int(tmp3), int(tmp4)),cv2.NORM_L2)
        print("Reprojection Error :",error)
        mean_error+=error

    mean_error=(mean_error)/float(len(tar_pt_3ds))
    mean_error = str(mean_error)
    image = cv2.putText(image,"Reprojection Error :"+mean_error, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.imshow('image', image)
    cv2.waitKey(0)