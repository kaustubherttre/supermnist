from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
(X_train,y_train),(X_test,y_test)= load_data()
X_train= X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2],1))
X_test= X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
input_shape= X_train.shape[1:]
classes = len(unique(y_train))
#print(input_shape,classes)
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer= 'glorot_uniform', activation= 'relu', input_shape= input_shape))
model.add(MaxPool2D(pool_size=(3,3), padding='valid'))
model.add(Flatten())
model.add(Dense(100,activation= 'relu', kernel_initializer= 'glorot_uniform'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train, epochs= 20, verbose=1,batch_size=128)
loss,acc=model.evaluate(X_test,y_test,verbose=1)
print(acc,loss)
#loss: 0.0257 - acc: 0.9918