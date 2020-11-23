import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Data(path):
    data = pd.read_csv(path)
    price = np.array([])
    
    zip = pd.get_dummies(data['zipcode'])
    data = data.join(zip)   

    #print(data)
    #data=data.drop(columns=['id'])
    #data=data.drop(columns=['id', 'zipcode'])
    #data=data.drop(columns=['id', 'zipcode', 'lat', 'long'])
    data=data.drop(columns=['id', 'zipcode', 'sale_yr', 'sale_month', 'sale_day', 'lat', 'long'])
    
    dataset=np.array(data)
    #print("dataset==",dataset.shape)
    
    if "price" in data.columns:
        price = dataset[:, 0]
        #print("Y:",Y)
        data=data.drop(columns=['price'])  #刪價錢，放到Y
        #print("data:",data)
        dataset=np.array(data)
    #print("dataset",dataset.shape)
    return dataset, price



def normalize(train,valid,test):
	tmp=train
	mean=tmp.mean(axis=0)    #算train平均
	std=tmp.std(axis=0)      #算train標準差

	train=(train-mean)/std
	valid=(valid-mean)/std
	test=(test-mean)/std
    
	return train,valid,test



X_train, Y_train = Data('./train-v3.csv')
#print("X_train",X_train)
#print("X_train",Y_train)
np.savetxt('./X_train.csv', X_train,delimiter=',', fmt='%i')

X_valid, Y_valid = Data('./valid-v3.csv')
#print("X_train",Y_valid)
X_test, Y_test = Data('C:./test-v3.csv')
#print("X_train",Y_test)



X_train,X_valid,X_test=normalize(X_train,X_valid,X_test)
print("X_vaild:",X_valid)



from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(40, input_dim=X_train.shape[1]),
    keras.layers.Dense(95, activation='relu'),

    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(2, activation='relu'),

    keras.layers.Dense(1)
])

model.compile(optimizer='adam',loss='mae')
history =model.fit(X_train, Y_train, batch_size=30, epochs=150, validation_data=(X_valid, Y_valid))

Y_predict = model.predict(X_test)


print("history:",history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss (Lower is better)')
plt.xlabel('Training Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('loss.png')
plt.show()





n = len(Y_predict) + 1
for i in range(1, n):
	b = np.arange(1, n, 1)
	b = np.transpose([b])
	Y = np.column_stack((b, Y_predict))

np.savetxt('./test.csv', Y, delimiter=',', fmt='%i')