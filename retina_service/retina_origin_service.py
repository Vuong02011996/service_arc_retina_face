from headbody.utils.tools import filter_predictions
import logging
import flask
import numpy as np
from face_recognition import models
from flask import Flask, jsonify
from flask_cors import CORS
from mxnet import ndarray as nd
from dotenv import load_dotenv
import os
from shm.reader import SharedMemoryFrameReader
from utils.common import img_to_array, logger_handler, preprocess_arcface


app = Flask(__name__)
CORS(app)


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

env_path = "../.env"
load_dotenv(dotenv_path=env_path)

dic_key = {}
config_path = "../configs/api.yaml"

if int(os.getenv("USE_RETINA")) == 1:
    print("Retina is starting")
    retina_model = models.build_model("retina-r50", config_path)


@app.route("/predict/retina", methods=["POST"])
def retina():
    """ Face Detection Endpoint
    Args:
        image: RGB
    Returns:

    """
    _default_detection_scale = [1]

    # get the data from request
    response = flask.request.form

    # we need upscale the bounding boxes
    original_width = int(response["ori_width"])
    original_height = int(response["ori_height"])
    detection_scale = response.get("detection_scale")
    rgb_image = img_to_array(flask.request.files["image"].read())
    # preprocessing the image
    # input : RGB
    net_data = rgb_image.transpose(2, 0, 1)
    net_data = np.expand_dims(net_data, axis=0)
    net_data = nd.array(net_data)

    # prediction
    boxes, points = retina_model.detect_fast(net_data, (original_height, original_width, 3), 0.8, [float(detection_scale)])
    # convert to list
    boxes_list = boxes.tolist()
    points_list = points.tolist()

    data = {
        "sucess": True,
        "boxes": boxes_list,
        "points": points_list
    }
    return jsonify(data)


if __name__ == "__main__":
    app.run(host=os.getenv("RETINA_HOST"), port=int(os.getenv("RETINA_PORT")), threaded=False, debug=False)
