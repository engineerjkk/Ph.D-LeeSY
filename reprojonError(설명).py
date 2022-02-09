#그후 이제 방금 보신 데이터를 활용해서 reprojection error를 확인해보겠습니다.
#먼저 필요한 라이브러리들을 import 해주겠습니다.
#openCV와 넘파이, Collection과 pathlib를 import 해줍니다.

import cv2
import os
import numpy as np
import collections
from pathlib import Path
from tqdm import tqdm
from hloc.utils.parsers import parse_image_lists, parse_retrieval, names_to_pair

# 각각의 데이터들을 저장시킬 변수를 만들어줍니다. 여기서 각 데이터들을 정렬해서 저장시켜줄 것입니다.
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
#그리고 우선 사용될 함수들을 설명드리겠습니다.

#이 함수는 쿼터니온을 로테이션매트릭스로 변환하여주는 함수입니다.
#향 후 현재 저장되어있는 데이터는 방향 데이터가 4차원의 쿼터니온으로 구성되어있지만,
#reprojection 시켜줄때 필요한 방향값은 rotation matrix이기 때문입니다.
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

#images.txt 파일을 불러와주는 함수이구요.
#images.txt에는 Image ID와 쿼터니온값, 그리고 위치값인 X,Y,Z와 카메라 ID와 이미지 이름이 함께 저장되어있는 데이터입니다.
#그리고 마지막으로 2D 좌표와 이에 해당하는 3D좌표가 담긴곳을 가리키는 iD역시 저장되어있습니다.
#향후 이 iD 값을 통해 3D 좌표를 불러올 것입니다.
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

#cameras.txt 파일 읽어오는 함수입니다.
#카메라 ID와 모델, 그리고 너비와 높이, 마지막으로 카메라 내부파리미터가 리스트형태로 저장되어있습니다.
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

#3D 포인트값있는 txt 파일 읽어와주는 함수입니다.
#앞서 images.txt 데이터에 저장된 iD값을 통해 해다 해당 데이터에서 3D 좌표값, x,y,z를 불러와줄 수 있습니다.
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



#우선 필요한 데이터셋들을 불러 와야 겠죠.
#각 이미지의 feature 포인트들의 좌표가 담긴 images.txt데이터와, 3D 포인트 데이터들이 담긴 points3D, 그리고 해당 이미지를 찍었던 카메라 내부파라미터값이 담긴 데이터 cameras.txt를 불러옵니다.
hloc_images = read_images_text('outputs/aachen/sfm_sift/images.txt')
hloc_points3Ds = read_points3D_text('outputs/aachen/sfm_sift/points3D.txt')
hloc_cameras = read_cameras_text('outputs/aachen/sfm_sift/cameras.txt')


#그리고 이번시간에 확인하고자 하는 image id를 keys 라는 리스트에다가 담아두겠습니다.
#이 id에 해당하는 이미지와 데이터들로 이번 실습을 진행하겠습니다.
#130,255,111,630 총 네장의 이미지의 reprojection error를 구해보겠습니다.
keys = [130,255,111,630]
#그리고 각 keys 리스트 안을 돌면서 처리해주겠습니다.
for i in range(len(keys)):
    key = keys[i]
    #우선 해당하는 id값을 통하여 해당하는 데이터를 불러와줍니다.
    tar_img = hloc_images[key]
    tar_cameras = hloc_cameras[key]

    #그리고 좀더 보기 쉽게하기 위하여, header 파일 형태로 list를 다시 정리하여 주겠습니다.
    tar_camera_header = [tar_cameras.id, tar_cameras.width, tar_cameras.height, *tar_cameras.params]
    tar_image_header = [tar_img.id, *tar_img.qvec, *tar_img.tvec, tar_img.camera_id, tar_img.name]

    #3D 좌표를 담을 리스트를 초기화하여 줍니다.
    #여기 리스트에 각 이미지에 해당하는 2D 좌표와 3D 좌표들의 id값을 담을 예정입니다.
    tar_points_strings = []

    #for문을 통하여 각 해당 이미지의 xt좌표와 해당 이미지의 3D 좌표 id를 반복해 불러오겠습니다.
    #현재의 for문은 해당 이미지 안에서 각 한개한개의 feature point좌표값에 해당하는 데이터를 불러와 하나의 리스트안에 저장하는 과정입니다.
    #그리고 헷갈리면 안되는 것이 현재 point3D_ids값은 3D좌표가 아닌 3D좌표를 가리키는 id입니다. 이 id에 해당하는 3D 좌표값들을 잠시 후 불러와 사용할 것입니다.
    for xy, point3D_id in zip(tar_img.xys, tar_img.point3D_ids):
        #여기서 데이터에 -1값이 있는데요 이값은 저희가 필요한 데이터가 아니므로 -1이 등장할 경우 무시하고 건너뜁니다. #-1보여주기
        if(point3D_id == -1):
            continue
        #그리고 append 를 통해 x,y좌표와 3D id값을 tar_points_string에 담아둡니다.
        tar_points_strings.append(" ".join(map(str, [*xy, point3D_id])))
    #(잠깐쉬고)
    #해당하는 이미지가 담긴 경로를 지정해주겠습니다.이미지들이 담긴 폴더 경로를 image_directory 변수에 저장합니다.
    image_dir =Path('datasets/aachen/images/images_upright/')
    #여기서 tar_image_header[9]가 의미하는것은 리스트의 마지막 값으로 이미지 이름을 나타내고 있습니다.  그래서 이 path에서 id에 해당하는 이미지를 불러와줄것입니다.
    path = image_dir / tar_image_header[9]
    #불러온 이미지를 변수 image에 저장합니다.
    image = cv2.imread(str(path))


    tar_pt_3ds = []
    ##################################
    mean_error=0
    #따라서 해당 이미지에서 tar_points_strings 값 하나씩 불러와주겠습니다.
    for str_points in tar_points_strings:
        #현재 str_points에는 3차원의 데이터가 저장되어있습니다.
        #X,Y좌표, 그리고 3D좌표값을 나타내는 id가 들어있죠.
        #각 해당하는 데이터들을 split함수를 통하여 쪼개주어 각각의 값들을 각각의 변수에 저장해주겠습니다.
        list_str = str_points.split(" ")
        #tmp1에는 x좌표값을 저장하며,
        tmp1 = float(list_str[0])
        #tmp2에는 y좌표값을 저장합니다.
        tmp2 = float(list_str[1])
        #그리고 마지막으로 point3D_id 좌표에는 3D 포인트 좌표에 대한 ID를 저장해두겠습니다.
        point3D_id = int(list_str[2])

        #이후 각 이미지가 가졌던 3D point id값을 통해 진짜 3D 좌표값들을 불러옵니다.
        tar_3dPt = hloc_points3Ds[point3D_id]

        #모든 3D 좌표들을 tar_pt_3ds에 저장해두겠습니다. 그러기 위해선 미리 초기화를 해둬야겠죠?(방향키위)
        #이걸 쓰는 이유는 추후 전체 포인트의 개수를 알기 위함입니다.
        tar_pt_3ds.append(tar_3dPt.xyz)

        #이후 X,Y 좌표를 print로 찍어서 확인해보겠습니다.
        print("x좌표",int(tmp1))
        print("y좌표",int(tmp2))
        #이제 불러온 이미지에 실제 ground truth값을 찍어보겠습니다. 해당 x,y좌표와 크기는1, 그리고 초록색으로 보여드리며 마지막 파라미터는 -1을 통해 내부를 꽉 차게 칠해줍니다.
        image=cv2.circle(image, (int(tmp1), int(tmp2)), 1, (0,255,0), -1)

        #(이미지 보여주기)
        #그럼 해당하는 이미지를 위처럼 보실 수 있는데요. 하지만 여기서 이제 3D좌표로부터 reprojection 시켜서 얼마나 에러가 있는지를 구해야합니다.
        #그래야 추가적으로 더 좋은 최적화 알고리즘을 통해 에러가 최소화하는 방향으로 최적화시킬 척도가 될 수 있습니다.

        #우선 아래와 같은 openCV에서 제공해주는 projectPoint함수가 있는데요. 내부 파라미터를 저희가 채워주어야합니다.
        #(projectionPoints 함수 보여주기)

        #우선 저희가 가지고있는 데이터는 쿼터니온으로 구성되어있습니다. projection해주는 파라미터는 rotation matrix로 구성되어있으므로 rotation matrix로 변환해주어야합니다.
        #따라서 우선 해당 데이터로부터 쿼터니온을 추출합니다.
        tar_quat = np.array(tar_image_header[1:5])#쿼터니온
        #그리고 쿼터니온을 로테이션 변환함수에 넣어 변환후 변수에 저장합니다.
        rvec=qvec2rotmat(tar_quat)
        #그후 위치를 나타내는 translation vector역시 변수에 저장합니다.
        tar_trans = np.array(tar_image_header[5:8])

        #이제 카메라 내부파라미터를 설정해주어야합니다.
        #우선 3by3행렬로 camera metrix를 초기화해주겠습니다.
        camera_metrix = np.zeros((3, 3), dtype='float32')
        #그리고 focalLenth를 저장해주고요.
        camera_metrix[0, 0] = tar_camera_header[3]  #focal_Length
        #1,1위치에도 focal length를 저장합니다.
        camera_metrix[1, 1] = tar_camera_header[3]
        #4,5번에는 각각의 주점을 저장합니다.
        camera_metrix[0, 2] = tar_camera_header[4]
        camera_metrix[1, 2] = tar_camera_header[5]
        #그리고 상수 1을 저장하며,
        camera_metrix[2, 2] = 1
        #왜곡계수를 설정합니다. 평소 왜곡계수는 0으로 하는경우가 많지만 해당 경우에는 왜곡계수역시 데이터로 주어졌으므로 사용하겠습니다.
        #왜곡계수를 distort 변수에 저장해줍니다.
        distort=tar_camera_header[6]

        #그리고 projectPoint 내부 파라미터를 채워주겠습니다.
        #3차원의 3D좌표, Rotation Matrix, Translation Vector, 그리고 카메라 내부파라미터와 왜곡계수인 Skew error값을 대입합니다.
        imgpoints2, _ = cv2.projectPoints(tar_3dPt.xyz, rvec, tar_trans, camera_metrix, distort)
        #해당값은 3차원이므로 1차원으로 만들어주기위해 차원을 두번 반복해 제거해주고요.
        imgpoints2 = np.concatenate(imgpoints2).tolist()
        imgpoints2 = np.concatenate(imgpoints2).tolist()

        #재사영된 2D 좌표를 변수에 저장해주겠습니다.
        tmp3 = float(imgpoints2[0])
        tmp4 = float(imgpoints2[1])

        #3D 좌표와 사영된 좌표를 확인하기 위해 출력해보겠습니다.
        print("3D 좌표 : ", *tar_3dPt.xyz)
        print("재사영된 2D x좌표 : ",int(tmp3))
        print("재사영된 2D y좌표 : ",int(tmp4))
        #그리고 이번엔 재사영된 좌표를 빨간색으로 해당 이미지에 찍어 비교해보도록하겠습니다.
        image = cv2.circle(image, (int(tmp3), int(tmp4)), 1, (0, 0, 255), -1)#reprojection 찍기

        #그리고 각각의 찍인 ground truth 좌표와 재사영된 좌표간의 오차도 구해야겠죠?
        #그래서 norm 함수를 통하여 실제값 좌표와 재사영된 좌표의 차를 절댓값으로 구해줍니다.
        error=cv2.norm((int(tmp1), int(tmp2)),(int(tmp3), int(tmp4)),cv2.NORM_L2)
        #그렇게 reprojection error값도 프린트해서 볼 수 있습니다.
        print("reprojection error :",error)
        #그리고 모든 좌표에대한 reprojection error 평균값을 구하기 위해 모든 error값들을 더해줍니다.
        mean_error+=error#에러는 길이단위가 아니라 픽셀단위이기때문에 상대적값으로 보아도 된다. 이 역시도 mean_error를 초기화 해두겠습니다.
    #그리고 모든 좌표 개수로 나누어주면 평균 에러가 나옵니다.
    mean_error=(mean_error)/float(len(tar_pt_3ds))
    # 이값을 string으로 변환하여 해당 이미지에 함께 보도록 출력해주겠습니다.
    mean_error = str(mean_error)
    #putText 함수를 통해 평균적인 reprojection error을 확인해보겠습니다. 위치는 왼쪽 상단, 그리고 색상은 파란색으로 보겠습니다.
    image = cv2.putText(image,"Reprojection Error :"+mean_error, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    #그리고 해당 이미지를 출력하여 보시면 다음과 같습니다.
    cv2.imshow('image', image)
    cv2.waitKey(0)
    #그럼 실행해보겠습니다.(끝)



#다소 시간이 걸려서 좀 기다리셔야합니다. 그 이유는 여기 image.txt파일과 points3d, camera.txt 파일이 용량이 꽤 됩니다.
#그래서 이걸 읽어와서 변수에 저장하는데 꽤 시간이 걸리는 이유가 되겠구요 잠시 기다려보겠습니다.

    #이와같이 초록색은 실제 feature point이며, 빨간색은 reprojection 된 포인트입니다.
    #그리고 왼쪽 상단에 평균 reprojection error을 보실 수 있습니다.
#우선 해당 이미지의 reprojection error는 0.87414 이구요 에러가 평균적으로 1픽셀도 되지않습니다.
#자세히 보시면 1픽셀 또는 2픽셀 에러가 난경우가 있구요 또는 이렇게 완전 정확하게 매칭되는것을 보실 수도 있습니다.
#그리고 feature point관점에서도 말씀을 드리면 3D 포인터를 생성할때 동적인 물체는 여러장의 이미지의 bundle adjustment과정에서 매칭이 되지 않기 때문에
#3D 포인트가 생성되지 않습니다. 그래서 여기 움직이는 사람의 경우 feature가 추출되지않구요. 그리고 지금 사진을 찍고있는 사람은 가만히 있죠. 그래서 feature point가 찍힘을 보실 수 있습니다.
#그리고 보통 바람이 불면 나무가 흔들려서 가변적이기 때문에 나무가지가 주변 gradient 값이 커도 feature point값이 잘 추출되지 않는데요. 해당 이미지의 경우 잘 추출되는것을 보아 현재
#바람이 불지 않나봅니다. 다음 이미지를 보시겠습니다.

#해당 이미지는 reprojection error가 평균 1.03이 찍힌것을 보실 수 있구요.
#여기서도 역시 움직이는 사람은 feature point가 추출되지 않는 반면 지금 구경하고계신 가만히있는 할머니는 feature point가 조금 추출됐습니다.
#그리고 건물 골고루 feature가 추출되엉있구요. 다음 이미지르 보시면,

#해당 이미지에서 reprojectio error는 0.866이 되겠구요.
#건물들은 feature가 잘 추출되어있고, 자세히 보시면 이렇게 reprojection 이 찍힌것을 보실 수 있습니다.
#왼쪽 나무는 feature가 잘 찍히지 않았구요, 오른쪽 나무는 featurepoint가 잘 찍혔네요. 그리고 바닥은 feature point가 잘 찍히지 않았음을 보실 수 있습니다.
# 바닥도 feature point가 찍힐법한데, 저자가 일부로 관심영역을 설정해준걸수도 있을것같네요. 마지막이미지를 보시겠습니다.

# 해당 이미지는 평균 reprojection error가 0.95이구요.
# 가까이 보시면 여기 딱 눈에 띄는 feature point역시도 에러없이 잘 매칭된것을 보실 수 있습니다.
# 해당 이미지에서는 거의 모든사람이 움직이는 동적인 물체라서 feature point들이 잘 찍히지않구요 여기 나뭇잎들도 바람에 날리는지 feature point역시 찍히지 않습니다.

#그리고 이처럼 실제 2D좌표는 1286,348이구요 여기 3D 좌표로부터 reprojection된 2D 좌표는 1287, 347이죠. 각각 1픽셀씩 차이가 나는데 그럼 대각선의 길이는 루트2인 1.414가 나오는 것을 볼 수 있습니다.
#reprojectin error는 이와같은 형태로 계산을 해주게 되구요. 이렇게 reprojection error를 정량적으로 구해놓아야. 이처럼 reprojectio error가 최소화하는 방향으로 리븐버그 마크워트와 같은
#최적화 알고리즘을 적용할 수있는 정량적 기준이 됩니다. 감사합니다.