#Credit: Shivam Patel, karl_cempron
import matplotlib.pyplot as plt #用plt.ginput()
from scipy.spatial import Delaunay #用三角剖分
from scipy.spatial import tsearch
import dlib # 用模型和auto-detect
import cv2
import imageio
import numpy as np

FRAME_COUNT = 45  # 45
FRAME_PER_SECOND = 30

FRAME_TO_PATH = "wu_to_su_"
GIF_SOURCE_PATH = FRAME_TO_PATH
IMG1 = "./input/wuyanzu.jpg"
IMG2 = "./input/sudaqiang.jpg"
IF_RECAL = False  # If not, import stored_points; if yes, go to IF_AUTO.
IF_AUTO = True  # If not, manually pin feature points, else, automatically detect feature points.
GIF_NAME = "./output/wu_to_su.gif"

'''Manual-way parameters'''
POINT_COUNT = 70  # Manual-way
CORNER_POINTS = [[0, 0], [0, 499], [299, 0], [0, 299], [299, 499],
                 [499, 299], [499, 0], [499, 499]] #在人工选点中加入8个样本点

'''Auto-way parameters'''
FEATURE_POINT_NUM = 68  # 待修改
TOTAL_POINT_NUM = FEATURE_POINT_NUM + 8
PREDICTOR = "./input/shape_predictor_68_face_landmarks.dat"

'''Manually find correspondences between two images.'''
def manuallyMakeCorrespondences(src, dest, filename):
    imA_points = findPoints(src)
    imB_points = findPoints(dest)
    f = open("./stored_{}points.py".format(filename), "w")
    f.write("imA_points = " + str(imA_points) + "\n")
    f.write("imB_points = " + str(imB_points) + "\n")
    f.close()
    print("Points saved!")
    return imA_points, imB_points

'''Helper function for ManuallyMakeCorrespondences.'''
def findPoints(img):
    i = 0
    plt.imshow(img)
    img_points = []
    while i < POINT_COUNT:
        x = plt.ginput(1, timeout=0)
        img_points.append([x[0][0], x[0][1]])
        plt.scatter(x[0][0], x[0][1])
        plt.draw()
        i += 1
    plt.close()
    for corner in CORNER_POINTS:
        img_points.append(corner)
    return img_points

'''Automatically detect feature points for two images.'''
def autoMakeCorrespondence(src, dest, filename):
    #Read images.
    if (isinstance(src, str)):
        img1 = cv2.imread(src)  # src为路径字符串，使用imread方法读取
    else:
        img1 = cv2.imdecode(np.fromstring(src.read(), np.uint8), 1)
    if (isinstance(dest, str)):
        img2 = cv2.imread(dest)
    else:
        img2 = cv2.imdecode(np.fromstring(dest.read(), np.uint8), 1)
    imgList = [img1, img2]

    #Initialize some values.
    j = 0
    list1 = []
    list2 = []

    # Detect points of the face.
    predictor_path = PREDICTOR
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for img in imgList:
        # size 存放tuple,两张照片的大小统一
        size = (img.shape[0], img.shape[1])
        if j == 0:
            currList = list1
        else:
            currList = list2
        detects = detector(img, 1)
        j += 1
        for k, face in enumerate(detects):
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, face)
            for i in range(0, 68):
                currList.append([int(shape.part(i).x), int(shape.part(i).y)])
            currList.append([0, 0])
            currList.append([size[1] - 1, 0])
            currList.append([(size[1] - 1) // 2, 0])
            currList.append([0, size[0] - 1])
            currList.append([0, (size[0] - 1) // 2])
            currList.append([(size[1] - 1) // 2, size[0] - 1])
            currList.append([size[1] - 1, size[0] - 1])
            currList.append([(size[1] - 1) // 2, (size[0] - 1) // 2])
    f = open("./stored_{}points.py".format(filename), "w")
    f.write("imA_points = " + str(list1) + "\n")
    f.write("imB_points = " + str(list2) + "\n")
    f.close()
    #Return the points detected for image_source and image_destination.
    return (list1, list2)

'''Do morphing on images.'''
def morph(imA, imB, imA_points, imB_points):
    weights = weightsMatrix()
    for i in range(FRAME_COUNT):
        mid_Points = findMidLocations(imA_points, imB_points, weights[i])
        delaunay_mid = Delaunay(mid_Points) #进行三角剖分
        frame = blendingfaces(imA, imB, imA_points, imB_points, mid_Points, delaunay_mid, weights[i])
        imageio.imwrite("./output/" +FRAME_TO_PATH + "{}.jpg".format(i), frame)
        print(i)

'''权重矩阵---Morph方法的help function 1'''
def weightsMatrix():
    return np.linspace(0.0, 1.0, FRAME_COUNT)

'''找到平均变换位置坐标---Morph方法的help function 2'''
def findMidLocations(src_points, dest_points, weight):
    locations = []
    for index in range(len(src_points)):
        src = src_points[index]
        dest = dest_points[index]
        x = (1 - weight) * src[0] + weight * dest[0]
        y = (1 - weight) * src[1] + weight * dest[1]
        locations.append([x, y])
    return np.array(locations)

'''对图像进行Blending---Morph方法的help function 3'''
def blendingfaces(src, dest, src_points, dest_points, imMid_points, delaunay_mid, weight):
    blendingMatrices1 = affineHelper(src_points, imMid_points, delaunay_mid)
    blendingMatrices2 = affineHelper(dest_points, imMid_points, delaunay_mid)
    morphedImage = np.zeros((src.shape[0], src.shape[1], 3), dtype="float32")
    for y in range(src.shape[0]):
        # print(imA.shape[0])
        for x in range(src.shape[1]):
            # print(imA.shape[1])
            index = tsearch(delaunay_mid, (x, y))
            blend_src = np.dot(np.linalg.inv(blendingMatrices1[index]), [x, y, 1])
            blend_dest = np.dot(np.linalg.inv(blendingMatrices2[index]), [x, y, 1])
            src_x = np.int(blend_src[0, 0])
            src_y = np.int(blend_src[0, 1])
            dest_x = np.int(blend_dest[0, 0])
            dest_y = np.int(blend_dest[0, 1])
            morphedImage[y, x, :] = src[src_y, src_x, :] * (1 - weight) + dest[dest_y, dest_x, :] * weight
    return morphedImage

'''Blendingfaces的help function 1'''
def affineHelper(image_points, mid_points, delaunay_mid):
    blendingMatrices = []
    for delaun in delaunay_mid.simplices:
        img = image_points[delaun,]
        mid = mid_points[delaun,]
        blendingMatrices.append(computeAffineMatrix(img, mid))
    return blendingMatrices

'''Blendingfaces的help function 2'''
def computeAffineMatrix(img, mid):
    value_matrix = np.matrix("{} {} 1 0 0 0;".format(img[0][0], img[0][1])
                + "0 0 0 {} {} 1;".format(img[0][0], img[0][1])
                + "{} {} 1 0 0 0;".format(img[1][0], img[1][1])
                + "0 0 0 {} {} 1;".format(img[1][0], img[1][1])
                + "{} {} 1 0 0 0;".format(img[2][0], img[2][1])
                + "0 0 0 {} {} 1".format(img[2][0], img[2][1]))
    b = np.matrix("{} {} {} {} {} {}".format(mid[0][0], mid[0][1], mid[1][0], mid[1][1], mid[2][0], mid[2][1]))
    affine_values = np.linalg.lstsq(value_matrix, np.transpose(b))[0]
    affine_Matrix = np.vstack((np.reshape(affine_values, (2, 3)), [0, 0, 1]))
    return affine_Matrix

#
# def drawTriangulatedGraph():
#     #展示
#     #保存图片

def compose_gif():
    gif_images = []
    for i in range(FRAME_COUNT):
        path = FRAME_TO_PATH + str(i) + '.jpg'
        gif_images.append(imageio.imread(path))
    imageio.mimsave("./output/" + GIF_NAME, gif_images, fps=FRAME_PER_SECOND)  # fps frames per second, a frame == 1/30 of a second

def main():
     ### Uncomment this section to apply morphing
    src = plt.imread(IMG1)
    dest = plt.imread(IMG2)
    if IF_RECAL:
        if IF_AUTO:
            #Automatically detect correspondences between two pictures.
            src_points, dest_points = autoMakeCorrespondence(IMG1, IMG2, FRAME_TO_PATH)
        else:
            #Manually find correspondences between two pictures.
            manuallyMakeCorrespondences(src, dest, FRAME_TO_PATH)
        morph(src, dest, np.array(src_points), np.array(dest_points))
    else:
        import stored_wu_to_su_points  # Change the name if necessary
        src_points = stored_wu_to_su_points.imA_points
        dest_points = stored_wu_to_su_points.imB_points
        morph(src, dest, np.array(src_points), np.array(dest_points))
    compose_gif()

main()
