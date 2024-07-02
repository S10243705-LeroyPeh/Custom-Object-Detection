import os
import time
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import streamlit as st
import tempfile

# Define flags
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

DEMO_VIDEO = 'demo_vid.mp4'

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def main(_argv):
    st.title('Compost Object Detection')
    st.sidebar.title('Compost Object Detection')

    use_webcam = st.sidebar.button('Use Webcam')
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.3)
    video_file_buffer = st.sidebar.file_uploader("Upload a video/image", type=["mp4", "mov", 'avi', 'asf', 'm4v', 'png', 'jpeg'])

    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    custom_classes = st.sidebar.checkbox('Use Custom Classes')
    if custom_classes:
        assigned_class = st.sidebar.multiselect('Select The Custom Classes', list(class_names.values()), default='car')

    stop_button = st.sidebar.button('Stop Processing')
    if stop_button:
        st.stop()

    stframe = st.empty()
    save_video = st.button('Save Results')

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'VP80')  # Corrected codec for webm
    out = cv2.VideoWriter('output.webm', codec, fps, (width, height))

    frame_num = 0

    while True:
        return_value, frame = vid.read()
        if not return_value:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
        image = Image.fromarray(frame)

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=confidence, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=confidence, input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=confidence
        )

        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        allowed_classes = list(class_names.values())
        if custom_classes:
            allowed_classes = assigned_class

        if FLAGS.crop:
            crop_rate = 150
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', 'video_name')
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)

        if FLAGS.count:
            counted_classes = count_objects(pred_bbox, by_class=False, allowed_classes=allowed_classes)
            for key, value in counted_classes.items():
                print(f"Number of {key}s: {value}")
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)

        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")
        result = np.asarray(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if save_video:
            out.write(result)

        result = image_resize(result, width=720)
        stframe.image(result, channels='BGR', use_column_width=True)

    st.success('Video saved')
    st.text('Video is Processed')
    
    output_vid = open('output.webm', 'rb')
    out_bytes = output_vid.read()
    
    st.text('Output Video')
    st.video(out_bytes)
    vid.release()
    out.release()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
