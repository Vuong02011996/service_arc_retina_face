import logging as log
import os
import shutil
from colorlog import ColoredFormatter
import cv2
import numpy as np
import math
from sklearn import preprocessing
from skimage import transform as trans
from PIL import Image
import io


def img_to_array(image):
    """ Preprocessing the image

    Args:
        image (BytesIO):

    Returns:
        image (np.array): HxWx3 RGB image
    """

    image = Image.open(io.BytesIO(image))
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = np.array(image)
    return image


def nparray_to_bytebuffer(image):
    """
    Convert the numpy array to bytes buffer
    It's used to send the np.array via request
    The server side can interpret the Io.Bytes Buffer
    Args:
        image (np.array): HxWx3

    Returns:
        byte_img (bytes): bytes image
    """
    if isinstance(image, np.ndarray):
        flag, frame = cv2.imencode(".jpg", image)
        if flag:
            frame = frame.tobytes()
            return frame
        else:
            raise RuntimeError("The image's format is not correct")
    else:
        raise RuntimeError("The data is not nd.array")


def logger_handler():
    """
    Logger Handler

    Return:
        A logger handler
    """
    # logging formatter
    formatter = ColoredFormatter(
        "%(asctime)s | %(bold_green)s%(name)s | %(log_color)s%(levelname)-8s%(reset)s %(message)s",
        datefmt="%d-%m %H:%M:%S",
        reset=True,
        log_colors={
            "INFO": "bold_green",
            "DEBUG": "bold_yellow",
            "WARNING": "bold_purple",
            "ERROR": "bold_red",
            "CRITICAL": "bold_red,bg_white",
        },
    )
    # define handler
    handler = log.StreamHandler()
    handler.setFormatter(formatter)
    return handler


def get_images_list(basePath, contains=None):
    """Function to return list of images in a folder

    Args:
        basePath
    Returns:

    """
    image_list = []
    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    # return the set of files that are valid
    image_list = list_files(basePath, validExts=image_types, contains=contains)
    return image_list


def list_files(basePath, validExts=None, contains=None):
    images = []
    # loop over the directory structure
    for (rootDir, _, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind(".") :].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                # yield imagePath
                images.append(imagePath)

    return images


def create_folders(basePath):
    """Create folders including main and sub folders

    Args:
        basePath (str): original path of folder
    """
    try:
        shutil.rmtree(basePath)
    except:
        pass
    os.makedirs(basePath)


def get_headpose(landmarks_2d, im_width, im_height):
    f = im_width
    u0, v0 = im_width / 2, im_height / 2
    camera_matrix = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]], dtype=np.double)

    landmarks_3d = np.array(
        [
            (0.0, 0.0, 0.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ]
    )

    dist_coeffs = np.zeros((4, 1))

    (_, rotation_vector, translation_vector) = cv2.solvePnP(
        landmarks_3d, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )

    return rotation_vector, translation_vector, camera_matrix, dist_coeffs


def get_roll_pitch_yaw(landmark5, im_width, im_height, MIN_ROLL, MIN_PITCH, MIN_YAW):
    """ Get angles of face """
    landmarks_2d = np.array(
        [
            (landmark5[2][0], landmark5[2][1]),
            (landmark5[0][0], landmark5[0][1]),
            (landmark5[1][0], landmark5[1][1]),
            (landmark5[3][0], landmark5[3][1]),
            (landmark5[4][0], landmark5[4][1]),
        ],
        dtype="double",
    )

    rotation_vector, translation_vector, _, _ = get_headpose(
        landmarks_2d, im_width, im_height
    )
    euler_angles = get_euler_angles(rotation_vector, translation_vector)

    if (
        (abs(euler_angles[0]) <= MIN_ROLL)
        and (abs(euler_angles[1]) <= MIN_PITCH)
        and (abs(euler_angles[2]) <= MIN_YAW)
    ):
        return [abs(euler_angles[0]), abs(euler_angles[1]), abs(euler_angles[2])]
    else:
        return None


def get_euler_angles(rvec, tvec):
    rmat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rmat, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(P)[6]

    pitch, yaw, roll = [math.radians(x) for x in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return [int(roll), int(pitch), int(yaw)]


def align_face(img, bbox, landmark):
    """Align the face based on the landmark and bounding boxes

    Args:
        img (np.array): raw image
        bbox (list): bounding box list
        landmark ():

    Returns:
        warped (np.array): aligned face with the shape of (112,112)
    """
    image_size = [112, 112]

    M = None
    if landmark is not None:
        src = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
    # M: The translation, rotation, and scaling matrix.
    if M is None:
        det = bbox
        margin = 44
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1] : bb[3], bb[0] : bb[2], :]

        ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:
        # do align using landmark
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped


def preprocess_arcface(image):
    """Preprocessing Step before using Arcnet Model

    Steps:
        - Convert BGR image into RGB
        - Tranpose the image

    Args:
        image (np.array): aligned face
    """
    nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nimg = np.transpose(nimg, (2, 0, 1))
    return nimg