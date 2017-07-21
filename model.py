import os
import csv
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Flatten, Dense, Lambda
import cv2
import numpy as np
import sklearn

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for line in batch_samples:
                source_path = line[0]
                #filename = source_path.split('\')[-1]
                #current_path = '.\\IMG\\' + filename
                #image = cv2.imread(current_path)
                image = cv2.imread(source_path)
                images.append(image)
                angles.append(float(line[3]))

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 65, 320  # Trimmed image format


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
print("model saved")
