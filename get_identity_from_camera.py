"""
Function1: điểm danh học sinh chấm công nhân viên.
Task1: Trích xuất identity từ camera.
    + Đọc camera trích xuất 10 frame hoặc 10 khuôn mặt tốt nhất.
    + Trích xuất đặc trưng 10 vector save to milvus.
    + TEST: Đưa một khuôn mặt tới camera xác định có nhận diện trong database hay không.
"""
import time

import kthread
import cv2
from queue import Queue
import argparse
import numpy as np
from headbody import models
from mxnet import ndarray as nd
from utils.common import (
    align_face,
    preprocess_arcface,
)
from face_recognition import models as model_face
from milvus_dal.face_dal import FaceDAL
import os
import json
import sys

face_dal = FaceDAL()

CAM_GG = {"360_cam": "rtsp://admin:Admin123@192.168.111.210/1", "floor_1": "rtsp://admin:abcd1234@192.168.111.212/1",
          "floor_2": "rtsp://admin:Admin123@192.168.111.211/1"}

path_data = "data/"
json_file = path_data + "person.json"
if not os.path.exists(path_data):
    os.makedirs(path_data)


def scale_image(image_rgb, scale_percent):
    width = int(image_rgb.shape[1] * scale_percent / 100)
    height = int(image_rgb.shape[0] * scale_percent / 100)
    dim = (width, height)
    scale_img = cv2.resize(
        image_rgb,
        dim,
        interpolation=cv2.INTER_CUBIC,
    )
    return scale_img


class InformationModel(object):
    def __init__(self):
        config_path = "configs/api.yaml"
        self.retina_model = model_face.build_model("retina-r50", config_path)
        self.arc_face = model_face.build_model("arc-face", config_path)


class InformationVideo(object):
    def __init__(self):
        self.input_path = '/home/vuong/Videos/VIDEOS/Test_model_head_body/trao_giai.mp4'
        self.camera = CAM_GG["360_cam"]
        self.cap = cv2.VideoCapture(self.camera)
        # self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("camera cannot open")
            exit(0)


def video_capture(image_queue, info_video):
    index_frame = 0
    while info_video.cap.isOpened():
        ret, image_raw = info_video.cap.read()
        if not ret:
            print("Camera not return frame")
            info_video.cap = cv2.VideoCapture(info_video.camera)
            # break
            continue
        image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image_queue.put([image_rgb, image_raw, index_frame])
        index_frame += 1
    info_video.cap.release()


def detect_face(image_queue, detections_queue, info_model, info_video):
    while info_video.cap.isOpened():
        image_rgb, image_raw, index_frame = image_queue.get()
        # img = scale_image(image_rgb, scale_percent=100)
        img = image_rgb
        net_data = img.transpose(2, 0, 1)
        net_data = np.expand_dims(net_data, axis=0)
        net_data = nd.array(net_data)
        start_time = time.time()
        face_boxes, points = info_model.retina_model.detect_fast(
            net_data, img.shape, 0.8, [1.0], do_flip=False
        )
        print("retina cost: ", time.time() - start_time)
        detections_queue.put([face_boxes, points, image_rgb, index_frame])
    info_video.cap.release()


def extract_embedding(detections_queue, embedding_queue, info_model, info_video, person_id, search=True):
    if os.path.exists(json_file):
        with open(json_file) as file:
            data_person = json.load(file)
    num_vector = 0
    pre_match = None

    while info_video.cap.isOpened():
        face_boxes, landmarks, image_rgb, index_frame = detections_queue.get()
        if len(face_boxes) > 0:
            print('len(face_boxes)', len(face_boxes))
            list_encoding_vector = []
            for i in range(len(face_boxes)):
                aligned_face = align_face(image_rgb, face_boxes[i][:4], landmarks[i])
                encoding_vector = info_model.arc_face.get_feature(preprocess_arcface(aligned_face))
                embedding_queue.put([encoding_vector, image_rgb, index_frame, num_vector])
                num_vector += 1
                list_encoding_vector.append(list(encoding_vector))

                box = list(map(int, face_boxes[i]))
                color = (0, 255, 0)
                cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), color, 2)

            if search:
                topk = 1
                start_time = time.time()
                list_results = face_dal.search_vector(list_encoding_vector, topk=topk)
                print("search cost ", time.time()-start_time)
                print(list_results)
                for i in range(topk):
                    # list_results[0] 1 vector search
                    dis = list_results[0][i]["dis"]
                    person_id = list_results[0][i]["person_id"]
                    print("similar: ", dis)
                    print("person_id: ", person_id)
                    print("name: ", data_person[str(person_id)])
                    if dis < 0.5:
                        print("------------------Matching------------------- ")
                        print("name: ", data_person[str(person_id)], "distance: ", dis)
                        if pre_match != data_person[str(person_id)]:
                            embedding_queue.put([-1, image_rgb, -1, -1])
                            pre_match = data_person[str(person_id)]
                            break

            else:
                list_person_id = [person_id] * len(list_encoding_vector)
                face_dal.insert_entities(list_encoding_vector, list_person_id)
        else:
            print("No face is detected")
            embedding_queue.put([None, image_rgb, index_frame, num_vector])
    info_video.cap.release()


def main(search):
    image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    embedding_queue = Queue(maxsize=1)
    info_video = InformationVideo()
    info_model = InformationModel()

    if not search:
        print("Enter your name: ")
        while True:
            name = input()
            if name:
                print('name', name)
                if os.path.exists(json_file):
                    with open(json_file) as file:
                        data_person = json.load(file)
                        person_id = int(list(data_person.keys())[-1]) + 1
                        data_person.update({str(person_id): str(name)})
                        print(data_person)
                else:
                    person_id = 0
                    data_person = {str(person_id): str(name)}
                # Save json
                with open(json_file, 'w+') as fp:
                    json.dump(data_person, fp)
                break
            else:
                print("Please enter your name. Enter your name: ")
    else:
        person_id = None

    t1 = kthread.KThread(target=video_capture, args=(image_queue, info_video))
    t2 = kthread.KThread(target=detect_face, args=(image_queue, detections_queue, info_model, info_video))
    t3 = kthread.KThread(target=extract_embedding, args=(detections_queue, embedding_queue, info_model, info_video,
                                                         person_id, search))
    t1.daemon = True
    t1.start()
    t2.daemon = True
    t2.start()
    t3.daemon = True
    t3.start()

    num_thread = []
    num_thread.append(t1)
    num_thread.append(t2)
    num_thread.append(t3)
    stop_all = False
    while True:
        if info_video.cap.isOpened():
            encoding_vector, image_rgb, index_frame, num_vector = embedding_queue.get()
            if num_vector == 1000 and not search:
                break
            if index_frame == -1:
                while True:
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    image_rgb = cv2.resize(image_rgb, (416, 416))
                    cv2.imshow('output_main', image_rgb)
                    if cv2.waitKey(10) & 0xFF == ord("c"):
                        break
                    if cv2.waitKey(10) & 0xFF == ord("s"):
                        stop_all = True
                        break
                if stop_all:
                    break
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            image_rgb = cv2.resize(image_rgb, (416, 416))
            cv2.imshow('output_main', image_rgb)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        else:
            break

    for t in num_thread:
        if t.isAlive() is True:
            t.terminate()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--search", type=bool, default=True,
                    help="search or add person to database, if search -s True")
    args = vars(ap.parse_args())
    main(search=args["search"])
    print("\n-------------End process-----------------")

    """
    Task
    + service model.
    + scale bbox follow detect face scale.
    + check matching fail when scale.
    """