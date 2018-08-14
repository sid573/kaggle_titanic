import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation


train_df = pd.read_csv('train1.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#print(train_df['Sex'])


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

 
np.random.seed(10)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)

model =Sequential()
model.add(Dense(units =4 ,input_dim=4, activation = 'relu'))
model.add(Dense(units = 12,activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train , Y_train,epochs = 50)

print(history)

Y_pred = model.predict_classes(X_test)
#Y_pred=np.round(Y_pred)
Y_pred = np.concatenate(Y_pred)
print (Y_pred.shape)


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
print(history.history.keys())
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.title('Training_accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.title('Loss')
plt.show()


submission.to_csv('submission.csv', index=False)
#print (train_df)

