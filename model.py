from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.preprocessing import image
import numpy as np
import keras

#First lets define a vision model using Sequential model
#This model will encode an image into a vector

img_width = 224
img_height = 224
num_channels = 3

vision_model = Sequential()
vision_model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(img_width, img_height, num_channels)))
vision_model.add(Conv2D(64, (3,3), activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3,3), activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3,3), activation='relu'))
vision_model.add(Conv2D(256, (3,3), activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Flatten())

#Now lets get a tensor with the output of our vision model
image_input = Input(shape=(img_width, img_height, num_channels))
encoded_image = vision_model(image_input)

#Next lets define a language model to encode the question into a vector
#Each question will be at most 100 words long
#and we will index words as integers from 1 to 9999
question_input = Input(shape=(2,), dtype='int32') #was 1000
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=2)(question_input)
encoded_question = LSTM(256)(embedded_question)

#Lets concatonate the question vector and the image vector
merged = keras.layers.concatenate([encoded_question, encoded_image])

#Lets train a logistic regression over 1000 words on top
output = Dense(2, activation='softmax')(merged) #was 1000

#This is our final model
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

img_path = 'C:\\Users\\DanJas\\Desktop\\IMG_-4tglqk.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

questions = ""

x2 = np.array([[1, 2]])
y = np.array([[1, 1]])

vqa_model.fit([x,x2], y)
