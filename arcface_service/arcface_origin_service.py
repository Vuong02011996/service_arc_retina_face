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

if int(os.getenv("USE_ARCFACE")) == 1:
    print("Arc_face is starting")
    arc_face_model = models.build_model("arc-face", config_path)


@app.route("/predict/arc_face", methods=["POST"])
def arc_face():
    aligned_face = img_to_array(flask.request.files["image"].read())
    encoding_vector = arc_face_model.get_feature(preprocess_arcface(aligned_face))
    assert isinstance(encoding_vector, np.ndarray)
    data = {"sucess": True, "encoding_vector": encoding_vector.tolist()}
    return jsonify(data)


if __name__ == "__main__":
    app.run(host=os.getenv("ARCFACE_HOST"), port=int(os.getenv("ARCFACE_PORT")), threaded=False, debug=False)
