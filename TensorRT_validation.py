import numpy as np
from onnx_helper import ONNXClassifierWrapper, ONNXClassifierWrapper2
import time
import cv2
import glob
import os
import albumentations as A
from algorithm import GramianAngularSumamationField, pre_processing

'''
This file run inference MICA-Net and validation on Jetson AGX Xavier 32 GB Board with UESTC-MMEA-CL dataset
'''


resolution = 172
transform = A.Compose([
    A.Resize(resolution, resolution, always_apply=True),
    A.ToFloat(),
])

PRECISION = np.float32
BATCH_SIZE = 1
print("Start load model")

# GAFormer
gaformer_feature_shape = 768
gaformer_input = np.zeros((BATCH_SIZE, 8, 224, 224), dtype=PRECISION)
trt_gaformer_model = ONNXClassifierWrapper("./UESTC-MMEA-CL/model/GAFormer_feature_extraction.trt", [BATCH_SIZE, gaformer_feature_shape], target_dtype=PRECISION)
G_feature = trt_gaformer_model.predict(gaformer_input)
G_feature = np.expand_dims(np.array([G_feature], dtype=PRECISION), axis=0)

# MoViNet
movinet_feature_shape = 600
movinet_input = np.zeros((BATCH_SIZE, 3, 16, 172, 172), dtype=PRECISION)
trt_movinet_model = ONNXClassifierWrapper("./UESTC-MMEA-CL/model/GAFormer_feature_extraction.trt", [BATCH_SIZE, movinet_feature_shape], target_dtype=PRECISION)
M_feature = trt_movinet_model.predict(movinet_input)
M_feature = np.expand_dims(np.array([M_feature], dtype=PRECISION), axis=0)

# MiCaNet
classes = 32
trt_MiCa_model = ONNXClassifierWrapper2("./UESTC-MMEA-CL/model/fusion_model.trt", [BATCH_SIZE, classes], target_dtype=PRECISION)
output = trt_MiCa_model.predict(G_feature, M_feature)

print("Start inference")
# Inference
data_fold = './UESTC-MMEA-CL/data/video/test'

count_true_prediction = 0
number_data_file = 0
for video_path in glob.glob(os.path.join(data_fold, '*', '*.mp4')):
    print(video_path)
    start = time.time()
    label = int(video_path.split("/")[-2]) - 1
    inertial_path = video_path.replace('video', 'inertial')[:-4] + '.csv'

    G_preprocess_start = time.time()
    gaformer_input = GramianAngularSumamationField(inertial_path)
    G_preprocess_end = time.time()
    print("Gaformer pre-process: ", G_preprocess_end - G_preprocess_start)

    M_preprocess_start = time.time()
    movinet_input = pre_processing(video_path, 12, transform)
    M_preprocess_end = time.time()
    print("MoViNet pre-process: ", M_preprocess_end - M_preprocess_start)

    G_predict_start = time.time()
    G_feature = trt_gaformer_model.predict(gaformer_input)
    G_predict_end = time.time()
    print("Gaformer predict: ", G_predict_end - G_predict_start)

    M_predict_start = time.time()
    M_feature = trt_movinet_model.predict(movinet_input)
    M_predict_end = time.time()
    print("MoViNet predict: ", M_predict_end - M_predict_start)

    G_feature = np.expand_dims(np.array([G_feature], dtype=PRECISION), axis=0)
    M_feature = np.expand_dims(np.array([M_feature], dtype=PRECISION), axis=0)

    fusion_start = time.time()
    output = trt_MiCa_model.predict(G_feature, M_feature)
    fusion_end = time.time()
    print("MiCaNet predict: ", fusion_end - fusion_start)

    classes = np.argmax(output)
    end = time.time()
    print("Total time with per gesture: ", end - start)
    if classes == label:
        count_true_prediction += 1
    number_data_file += 1

print("Number of true prediction: ", count_true_prediction)
print("Number of data file: ", number_data_file)
print(f"Accuracy: {count_true_prediction / number_data_file * 100} %", )
