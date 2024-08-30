from keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image   #for preprocessing the test images

#loading the training model file from the dir

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Model loaded from disk")


def classify(img_file):
    img_name = img_file     
    test_image = image.load_img(img_name, target_size = (64,64))    #preprocessing
    test_image = image.img_to_array(test_image)     #converting the img into array
    test_image = np.expand_dims(test_image, axis=0)     #expanding the dimensions of the testing image
    result = model.predict(test_image)

    if result[0][0] == 0:
        prediction= "Joker"
    else:
        prediction = "Harley Quinn"
    print(prediction, img_name)


import os
path = "Dataset\Test Data"
files = []

# r=root, d=dir, f=files
#iteration - specifying a folder, inside a subfolder, inside a file

for r, d, f in os.walk(path):
    for file in f:
        if ".jpg" in file: #going to pic an image from the file of either .jpeg or .jpg format
            files.append(os.path.join(r,file))  #appending the image in the array files

for f in files:
    classify(f) #passing the image to the classify function for testing
    print("\n")
