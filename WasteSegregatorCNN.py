    # Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a fourth convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())   
# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.1))

classifier.add(Dense(units = 2, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset-resized/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 20,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset-resized/test_set',
                                            target_size = (128, 128),
                                            batch_size = 20,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 40,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = 8)

# Part 3 - Evaluating The Model
import numpy as np
from keras.preprocessing import image

def evaluate(link, wasteType, n):
    
    missClassifications = 0
    plastic = 0
    glass = 0
    others = 0
    semiplastic = 0
    semiglass = 0

    for i in range(1,n):
        count1 = str(i)
        link += count1 + ".jpg"

        test_image = image.load_img(link, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)#4th dimension = batchs(requirement of predict function)
        result = classifier.predict(test_image)
        if result[0][1] == 0:
            prediction = 'glass'
            glass += 1
        elif result[0][0] == 0:
            prediction = 'plastic'
            plastic += 1
        elif result[0][0] > 0.70:
            prediction = 'semiglass'
            semiglass += 1
        elif result[0][1] > 0.70:
            prediction = 'semiplastic'
            semiplastic += 1
        else:
            prediction = 'others'
            others += 1
        if prediction != wasteType and prediction != "semi"+wasteType:
            missClassifications += 1
        if(i<10):
            link = link[:-5]
        elif(i<100):
            link = link[:-6]
        else:
            link = link[:-7]
            
    print(missClassifications)
    print("others = " + str(others)  + " plastic = " + str(plastic) + " glass = " + str(glass) + " semiPlastic " + str(semiplastic) + " semiGlass " + str(semiglass))
    training_set.class_indices
    count1 = 0
    if wasteType == 'plastic':
        count1 = glass
        count2 = semiglass
        count3 = others
    elif wasteType == 'glass':
        count1 = plastic
        count2 = semiplastic
        count3 = others
    else:
        count1 = (plastic+glass+semiplastic+semiglass) #assuming only 30% non-(plastics and glass) pass the sensors.
        count2 = 0
        count3 = 0
       
    print("efficiency = ")
    efficiency = 100*(1-((count1+(count2+count3))/n))
    print(efficiency)
    return efficiency
    
        

link = "dataset-resized/glass/glass"
n = 482 
n1 = evaluate(link, "glass", n)
link = "dataset-resized/plastic/plastic"
n2 = evaluate(link, "plastic", n)
link = "dataset-resized/paper/paper"
n3 = evaluate(link, "others", n)
link = "dataset-resized/metal/metal"
n = 350
n4 = evaluate(link, "others", n)

efficiency = (n1+n2+0.1*(n3+n4))/2.2
print("overall efficiency = " + str(efficiency))

    