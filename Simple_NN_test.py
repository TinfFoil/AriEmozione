from Tokenize_Vectorize import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation

#build the model
model=Sequential()
model.add(Dense(64,input_dim=32, activation='relu'))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(7,activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(aria_svd, dummy_y, nb_epoch=7, batch_size=32, verbose=1)

score = model.evaluate(dev_svd, dummy_dev, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


