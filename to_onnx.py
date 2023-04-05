import tensorflow as tf
from tensorflow import keras
import os

model = keras.models.load_model("pretrained/nsfw_mobilenet2.224x224.h5")
tf.saved_model.save(model, "tmp_model")
os.sys('python -m tf2onnx.convert --saved-model tmp_model --output "model.onnx"')