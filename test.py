import os
import numpy as np
from nsfw_detector import predict

model = predict.load_model('./pretrained/nsfw_mobilenet2.224x224.h5')

directory_path = "./images/"

# Predict for all images in a directory
for root, dirs, files in os.walk(directory_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        result_dic = predict.classify(model, file_path)[file_path]
        result_key = max(result_dic, key=result_dic.get)

        print(f'{file_path.replace(directory_path, "")}: {result_key} {(result_dic[result_key])*100}%')