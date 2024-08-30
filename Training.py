from keras.models import Sequential  #for sequencing the layers of DNN
from keras.layers import Conv2D     #Convolutional 2D network
from keras.layers import MaxPooling2D   #for maximum pooling
from keras.layers import Flatten    #for flattening the img
from keras.layers import Dense  #dense network - contains more than on hidden layer
from keras.preprocessing.image import ImageDataGenerator    #for preprocessing the img

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation = 'relu'))   #adding the first convolutional layer, (3,3) - kernel
model.add(MaxPooling2D(pool_size = (2,2)))  #performing the maximum pooling to the output of convolutional layer - maxPooling layer, (2,2)-kernel
model.add(Flatten())    #Performing the flattening on the output of maxPool
model.add(Dense(units = 128, activation = 'relu'))  #adding hidden layer with relu activationFn
model.add(Dense(units = 1, activation = 'sigmoid')) #adding hidden layer with sigmoid activation Fn(either 0 or 1 as o/p)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1/255, shear_range = 0.2, horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1/255)

#Pre-processing for Train and Validate Dataset
train_set = train_datagen.flow_from_directory("Dataset\Train Data" , target_size = (64,64), batch_size = 8, class_mode = "binary")
val_set = val_datagen.flow_from_directory("Dataset\Model Validation" , target_size = (64,64), batch_size = 8, class_mode = "binary")

#Training the dataset
model.fit_generator(train_set, steps_per_epoch = 50, epochs = 150, validation_data = val_set, validation_steps = 2)

#saving the model as the json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
