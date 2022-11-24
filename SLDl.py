import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np
import sys

def GetData(dataType):
    #get MNIST
    mnist = tf.keras.datasets.mnist
    (trainingData, trainingLabels), (testingData, testingLabels) = mnist.load_data()
    trainingData, testingData = trainingData / 255.0, testingData / 255.0
    if dataType == "train":
        return trainingData, trainingLabels
    elif dataType == "test":
        return testingData, testingLabels
    else:
        return trainingData, trainingLabels, testingData, testingLabels

def CustomModel(y_true, y_pred):
    return tf.keras.losses.MAE(y_true, y_pred)
    
def MakeModel(custom=False):
    #make a sequential model (stack of layers in order)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu',input_shape=(28,28,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    if custom == True:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001), #set optimizer to Adam; for now know that optimizers help minimize loss (how to change weights)
            loss=CustomModel, #sparce categorical cross entropy (measure predicted dist vs. actual)
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], #how often do predictions match labels
        )
    else:
        #configure model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001), #set optimizer to Adam; for now know that optimizers help minimize loss (how to change weights)
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), #sparce categorical cross entropy (measure predicted dist vs. actual)
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], #how often do predictions match labels
        )
    return model

def Predict(model, testingData, testingLabels):
    #predict and format output to use with sklearn
    predict = model.predict(testingData)
    predict = np.argmax(predict, axis=1)
    #macro precision and recall
    precisionMacro = precision_score(testingLabels, predict, average='macro')
    recallMacro = recall_score(testingLabels, predict, average='macro')
    #micro precision and recall
    precisionMicro = precision_score(testingLabels, predict, average='micro')
    recallMicro = recall_score(testingLabels, predict, average='micro')
    confMat = confusion_matrix(testingLabels, predict)

    print("Macro precision: ", precisionMacro)
    print("Micro precision: ", precisionMicro)
    print("Macro recall: ", recallMacro)
    print("Micro recall: ", recallMicro)
    print(confMat)

def Train(name):
    trainingData, trainingLabels = GetData("train")
    model = MakeModel()
    model.fit(trainingData, trainingLabels, epochs=1)
    model.save("./models/"+name+".h5")
    print("Model saved.")

def TrainBest(name, custom=False):
    checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath="./models/"+name+".h5", 
                             monitor='loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min'),
                tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.01, patience=2)]
                
    trainingData, trainingLabels = GetData("train")
    model = MakeModel(custom)
    model.fit(trainingData, trainingLabels, epochs=50, callbacks=checkpoint)


def Test(name):
    print("Loading Test Data")
    testingData, testingLabels = GetData("test")
    print("Loading model")
    model = tf.keras.models.load_model("./models/"+name+".h5")
    print("Making predictions on test data")
    Predict(model, testingData, testingLabels)

def CustomTrain(name):
    TrainBest(name, True)

def main():
    if sys.argv[1] == "-train":
        Train(sys.argv[2])
    elif sys.argv[1] == "-trainBest":
        TrainBest(sys.argv[2])
    elif sys.argv[1] == "-test":
        Test(sys.argv[2])
    elif sys.argv[1] == "-custom":
        CustomTrain(sys.argv[2])
if __name__ == "__main__":
    main()