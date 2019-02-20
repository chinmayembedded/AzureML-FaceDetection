import json
import numpy as np
import os
import tensorflow as tf

import base64
import io
import cv2
import base64 
from PIL import Image

from azureml.core.model import Model

NUM_CLASSES = 2
category_index = {2: {'id': 2, 'name': 'background'}, 1: {'id': 1, 'name': 'face'}}
detection_graph = tf.Graph()
sess = None
def init():
    global sess
    model_root = Model.get_model_path('frmodel')
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_root, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    with detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(graph=detection_graph, config=config)
    
def run(image):
    str = json.loads(image)['data']
    imgdata = base64.b64decode(str)
    data = Image.open(io.BytesIO(imgdata))
    image = cv2.cvtColor(np.array(data), cv2.COLOR_BGR2RGB)
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
    result = json.dumps({"boxes" : boxes.tolist(), "classes" : classes.tolist(), "scores": scores.tolist(), "num_detections": num_detections.tolist()})
    return result
    