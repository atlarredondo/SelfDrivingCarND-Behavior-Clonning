# coding: utf-8

# In[5]:

import csv
from scipy import misc
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Input, merge
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn


# In[9]:

def get_image(img_path, folder):
    path = get_image_path(img_path, folder)
    return misc.imread(path)

def get_image_path(img_path, folder):
    ''' Returns the image relative path
    '''
    return './%s/IMG/%s' % (folder, img_path.split('/')[-1])

def get_csv_lines(folder, flipped=False):
    '''
    extracts image path data and angle from csv file
    '''
    lines = []
    with open('./%s/driving_log.csv' % folder, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[0] = get_image_path(line[0], folder)
            line[1] = get_image_path(line[1], folder)
            line[2] = get_image_path(line[2], folder)
            line.append(flipped)
            lines.append(line)
    
    return lines

def get_data(folder):
    '''extracts images data and measurements by providing a folder'''
    lines = get_csv_lines(folder)

    center_images = []
    left_images = []
    right_images = []
    measurements = []

    for line in lines:
        center_images.append(get_image(line[0], folder))
        left_images.append(get_image(line[1], folder))
        right_images.append(get_image(line[2], folder))

        measurements.append(float(line[3]))
    
    return center_images, left_images, right_images, measurements


# In[10]:

def generator(samples, batch_size=32):
#     generator use to extract images in batches
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Reading all images from path
                center_image = misc.imread(batch_sample[0])
                left_image = misc.imread(batch_sample[1])
                right_image = misc.imread(batch_sample[2])
                
                # Reading angle and calculating left and right steering angles
                center_angle = float(batch_sample[3])
                steering_left = center_angle + correction
                steering_right = center_angle - correction
                
                # Append images to batch array by flipping it if specified
                if batch_sample[-1]:
                    images.extend([np.fliplr(center_image), np.fliplr(left_image), np.fliplr(right_image)])
                    angles.extend([-center_angle, -steering_left, -steering_right])
                else:
                    images.extend([center_image, left_image, right_image])
                    angles.extend([center_angle, steering_left, steering_right])
               
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[12]:

# Extracting all image path data and steering angle from 
csv_lines = get_csv_lines('data_f') + get_csv_lines('data_f_rev')  + get_csv_lines('data_f', True) + get_csv_lines('data_f_rev', True)
train_samples, validation_samples = train_test_split(csv_lines, test_size=0.2)

# Note: the csv files were read multiple times to keep data shuffled when spliting it between test/valid set
# and reading batches.

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[13]:

# Commented code that extracts the images before implementing generator
# center_images, left_images, right_images, measurements = get_data('data')
# center_images_rev, left_images_rev, right_images_rev, measurements_rev = get_data('data_rev')

# X_train = np.array(center_images + center_images_rev)
# y_train = np.array(measurements + measurements_rev)


# In[14]:

def converter(x):
    #RGB to greyscale converter
    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])


# In[15]:

def get_inception_module(prev):
    '''Creates an inception module given an input later'''
    tower_1 = Convolution2D(32, 1,1, border_mode='same', activation='relu')(prev)
    tower_1 = Convolution2D(16, 3,3, border_mode='same', activation='relu')(tower_1)
    
    tower_2 = Convolution2D(32, 1,1, border_mode='same', activation='relu')(prev)
    tower_2 = Convolution2D(16, 5,5, border_mode='same', activation='relu')(tower_2)
    
    tower_3 = MaxPooling2D((3,3), strides=(1,1), border_mode='same')(prev)
    tower_3 = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(tower_3)
    
    return merge([tower_1, tower_2, tower_3],  mode='concat', concat_axis=1)


# In[26]:

# Building the model:
dropout_prob = 0.5
input_img = Input(shape = (160, 320, 3))

# Data Preprocessing steps
p = Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))(input_img)
# p = Lambda(converter, input_shape=(160,320,3))(p)
p = Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320))(p)

# Convolutional layers
c1 = Convolution2D(24, 5, 5, subsample=(2,2), activation="relu")(p)
c2 = Convolution2D(36, 5, 5, subsample=(2,2), activation="relu")(c1)
c3 = Convolution2D(48, 5, 5, subsample=(2,2), activation="relu")(c2)

# Final convolutional layers with dropout
c4 = Convolution2D(64, 3, 3, activation="relu")(c3)
c4 = Dropout(dropout_prob)(c4)
c4 = Convolution2D(64, 3, 3, activation="relu")(c4)
c4 = Dropout(dropout_prob)(c4)

# Inception module(no longer used)
# i1 = get_inception_module(c2)
# inception_reduced = Convolution2D(32, "1,1, activation='relu')(d1)

# Flatten and dense layers
f = Flatten()(c4)

c = Dense(1164)(f)
c = Dropout(dropout_prob)(c)
c = Dense(200)(c)
c = Dense(50)(c)
c = Dense(10)(c)
o = Dense(1)(c)

model = Model(input=input_img, output=o)

model.compile(loss='mse', optimizer='adam')
print(model.summary())


# In[ ]:

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples*3),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples*3),
                                     nb_epoch=10,
                                     verbose=1
                                    )
model.save('model.h5')


# In[25]:


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()