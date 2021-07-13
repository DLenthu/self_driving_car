from utils import *
from sklearn.model_selection import train_test_split

path = "/home/bhargav/Documents/SELF_DRIVING/data"

data = importDataInfo(path)

data = balanceData(data,display = False)

imagesPath,steering = loadData(path,data)

xTrain ,xVal ,yTrain ,yVal = train_test_split(imagesPath,steering,test_size = 0.2,random_state = 5)

model = createModel()
model.summary()

history = model.fit(batchGen(xTrain,yTrain,25,1),steps_per_epoch = 20,epochs = 10,
        validation_data = batchGen(xVal,yVal,10,0),validation_steps = 20) 


model.save("model.h5")
print("Model is saved") 

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Training","Validation"])
plt.show()

 




