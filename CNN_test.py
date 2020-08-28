
# Padding/truncating the data (if necessary)
from Tokenize_Vectorize import *
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import train_test_split

aria_svd_x, aria_svd_y, dummy_y_x, dummy_y_y = train_test_split(aria_svd, dummy_y,
                                                                test_size=0.2)  # here I am just splitting the training dataset


# to mimic


embedding_dims = 32
filters = 250  # (!)
kernel_size = 3
hidden_dims1 = 250


#x_train = pad_trunc(aria_svd_x, max_len)
#x_test = pad_trunc(aria_svd_y, max_len)

x_train = np.expand_dims(aria_svd, axis=2)
y_train = np.array(dummy_y)
x_test = np.expand_dims(dev_svd, axis=2)
y_test = np.array(dummy_dev)
print(x_train.shape)




print('Building model...')
model = Sequential()  # The standard NN model
model.add(Conv1D(  # Adding a convolutional layer
    filters,
    kernel_size,
    padding='valid',  # in Finding_the_best_NeuronCombos.py example the output is going to be lightly smaller
    activation='relu',
    strides=1,  # the shift
    input_shape=(embedding_dims, 1))
)
model.add(GlobalMaxPooling1D(5))
model.add(Dropout(0.2))
model.add(Dense(hidden_dims1))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(6, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


