from nsfw_detector import predict
model = predict.load_model('./pretrained/nsfw_mobilenet2.224x224.h5')

# Predict for all images in a directory
predict.classify(model, './images')