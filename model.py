# import all the necessary packages.

import csv
import numpy as np
import cv2
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt
from collections import Counter
from keras.models import Sequential
from keras.layers import Flatten,Dense,Convolution2D,MaxPooling2D,Activation,Lambda,Cropping2D

#works on ipynb. Uncomment if running there
#%matplotlib inline



# Method Definitions:


# The below method is used to remove images of a particular steering angle which has more than 300 images and which has less than 10 images.
def Reduce_Extra_Images(input_x,input_y):
    augmented_image_data=[]
    augmented_steering_measures=[]
    original_count = Counter(input_y)
    for actual_image,steer in tqdm(zip(input_x,input_y)):
        count = Counter(augmented_steering_measures)
        #print(count)
        if(count[steer]<=300 and original_count[steer]>10):
            augmented_image_data.append(actual_image)
            augmented_steering_measures.append(steer)
        else:
            continue
        del count
    #print(len(count))
    return np.array(augmented_image_data),np.array(augmented_steering_measures)


	
# Augmentation method used to flip and adjust brightness to the image randomly. 
#This will also serve the same purpose as mentioned above with maximum images margin of 400.
# Not used Now.

def Augment(input_x,input_y):
    augmented_image_data=[]
    augmented_steering_measures=[]
    original_count = Counter(input_y)
    for actual_image,steer in tqdm(zip(input_x,input_y)):
        count = Counter(augmented_steering_measures)
        #print(count)
        if(count[steer]<=400 and original_count[steer]>10):
            augmented_image_data.append(actual_image)
            augmented_steering_measures.append(steer)
            if(steer!=0):
                rand = np.random.uniform(10,20)
                if(rand>15):
                    augmented_image_data.append(cv2.flip(actual_image,1))
                    augmented_steering_measures.append(steer*(-1.0))
                else:
                    augmented_image_data.append(bright_image(actual_image))
                    augmented_steering_measures.append(steer)
        del count
    #print(len(count))
    return np.array(augmented_image_data),np.array(augmented_steering_measures)   


#Adjusts Brightness of the image inputted to a random value added to 0.5 and returns the new image.
#Not used Now.
def bright_image(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1



#Actual Coding Starts here.
	
	
#initialize all the necessary variables

lines = []
images = []
steering_angles = []



# The below module is used to read the path of the images from an excel and then shuffle them.

with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

shuffle(lines)

# The below code loads all the images to the lists and converts them to numpy arrays.
for line in tqdm(lines):
    
    #The below is used to change the path of the file relatively
	
    file_name=line[0].split('/')[-1]
    #print(file_name)
    filepath = "./data/IMG/"+file_name
    #print(filepath)
    #break
    center_image = cv2.imread(filepath)
    center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
    images.append(center_image)
    center_angle = float(line[3])
    steering_angles.append(center_angle)
    
    left_file_name=line[1].split('/')[-1]
    #print(left_file_name)
    left_file_path="./data/IMG/"+left_file_name
    #print(left_file_path)

    left_image = cv2.imread(left_file_path)
    left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
    images.append(left_image)
    left_angle = center_angle + 0.20
    steering_angles.append(left_angle)
    
    right_file_name=line[2].split('/')[-1]
    #print(right_file_name)
    right_file_path="./data/IMG/"+right_file_name
    #print(right_file_path)
    
    #break
    
   
    right_image = cv2.imread(right_file_path)
    right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
    images.append(right_image)
    right_angle = center_angle - 0.20
    steering_angles.append(right_angle)
	

X_train = np.array(images)
y_train = np.array(steering_angles)

#delete the variables that are not needed anymore to free the memory.
del images
del steering_angles
del lines


#The below prints the shape of the Images Numpy Array.
print("The shape before removing images",X_train.shape)

print("Plotting initial data before removing images against each of the steering angles to get an intuition about the data")

#works on ipynb. Uncomment if running there
#plt.hist(y_train, bins=len(np.unique(y_train)), color='Red')


#Calling Reduce_Extra_Images method to remove extra images.
X_train,y_train = Reduce_Extra_Images(X_train,y_train)
print("The shape after removing images",X_train.shape)

print("Plotting after removing extra data against each of the steering angles to get an intuition about the data")

#works on ipynb. Uncomment if running there
#plt.hist(y_train, bins=len(np.unique(y_train)), color='Red')

print("See below to find how the steering angles are spread across the data set")

#works on ipynb. Uncomment if running there
#plt.plot(y_train)


# The Model Starts here.

# We use Keras to build the model over tensor flow, which works on back end.

model = Sequential()

#Cropping the image. Top 55 pixel rows and bottom 20 pixel rows.
model.add(Cropping2D(cropping=((55,20), (0,0)),input_shape=(160, 320, 3)))

#Normalizing using lambda. To make the value lie between 0 and 1. And subtracting 0.5 to make the mean 0.
model.add(Lambda(lambda x: (x/255.0) - 0.5))

#Adding the convolution and dense layers.
model.add(Convolution2D(6, 3, 3))
model.add(MaxPooling2D((2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D((2,2)))
model.add(Activation('elu'))

model.add(Flatten())

model.add(Dense(120))
model.add(Activation('elu'))

model.add(Dense(1))

#Using Mean Squared Error as loss function for regression problem. 
#Using adam optimizer. The learning rate will be chosen by the system itself.
model.compile(loss='mse',optimizer='adam')

#Shuffling the data before training.
#20% of the data set is chosen as validation data to check the effectiveness if the model and also to have a check on over and under fitting.
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

#Saving the model to run the simulator.
model.save('model.h5')

#To print the summary of the model we use.
model.summary()