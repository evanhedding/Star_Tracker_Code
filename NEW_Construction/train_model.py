import numpy as np
import csv
import pickle
import tensorflow as tf

with open("./dictionaries_36/guidestarDict_with_features", "rb") as fp:
    star_dict = pickle.load(fp)
    
with open("./dictionaries_36/id2label", "rb") as fp:
    id2label_dict = pickle.load(fp)
    
maxList = []
maxNum = 0
minNum = 10000
for guidestar_id in star_dict:
    if len(star_dict[guidestar_id]) > maxNum:
        maxNum = len(star_dict[guidestar_id])
        maxList = np.sort(np.asarray(star_dict[guidestar_id])[:,0])
    if len(star_dict[guidestar_id]) < minNum:
        minNum = len(star_dict[guidestar_id])
        
print(maxNum)
print(minNum)

########################################################################

print("Step 1 done.")
num_stars = len(star_dict)
train_data = []
test_data = []
labels = []
test_labels = []
val_labels = []
val_data = []
n = 0
newMax = 0
for star in star_dict:
    if n > 100:
        break
    
    if len(star_dict[star]) > newMax:
        newMax = len(star_dict[star])
        maxNum = len(star_dict[star])
        maxList = np.sort(np.asarray(star_dict[star])[:,0])
        
    label = id2label_dict[star]
        
        #Build test data
        #for k in range(15):
        #    idx = np.random.randint(0, len(star_dict[star]))
        #    val_data.append(star_dict[star].pop(idx))
        #    val_labels.append(label)
        
    #for i in range(len(star_dict[star])):
    for i in range(20):
        train_data.append(star_dict[star][i])
        labels.append(label)
    #print(n)
    n += 1
        
print("newMax", newMax)
print(maxList)

print("train samples:", len(train_data))
print("train labels:", len(labels))
print("validation samples:", len(val_data))
print("validation labels:", len(val_labels))
print("test samples:", len(test_data))
print("test labels:", len(test_labels))
print("Step 2 done.")
#Stores the test data and cooresponding labels
with open("./test_samples/test_data.txt", "wb") as fp:
    pickle.dump(test_data, fp)

with open("./test_samples/test_labels.txt", "wb") as fp:
    pickle.dump(test_labels, fp)

print("Step 3 done.")
#val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).shuffle(10000).batch(100)
print("Step 4.1 done.")
train_ds = tf.data.Dataset.from_tensor_slices((train_data, labels)).batch(1).repeat()
print("Step 4.2 done.")

########### Buckets for th1 #############################

thetaDict = {}
for i in range(maxNum + 1):
    #key = tf.cast(i, dtype=tf.float32)
    if i == 0:
        thetaDict[i] = (0., maxList[i])
    elif i == maxNum:
        thetaDict[i] = (maxList[i-1], 2*np.pi)
    else:
        thetaDict[i] = (maxList[i-1], maxList[i])
        

###################################################

class model_1(tf.keras.Model):
    def __init__(self):
        super(model_1, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.05)
        #self.drop = tf.keras.layers.GaussianDropout(0.05)
        
        self.n1 = tf.keras.layers.BatchNormalization()
        
        self.layersDict ={}
        
        for i in range(maxNum + 1):
            self.layersDict[i] = tf.keras.layers.Dense(num_stars, activation='softmax')
            # *May need to add more layers*
     
        ###############################################
        
    
    def call(self, x):
        th1 = tf.cast(x[:,0], dtype=tf.float32)
        
        
        #def truth(th1):
       #     if tf.math.logical_and(tf.math.greater(th1, thetaDict[i][0]), tf.math.less_equal(th1, thetaDict[i][1])):
        #        return True
       #     else:
        #        return False
        
        condList = []
        for i in range(len(thetaDict)):
            condList.append(tf.math.logical_and(tf.math.greater(th1, thetaDict[i][0]), tf.math.less_equal(th1, thetaDict[i][1])))
            
        def findLayer(i, x):
            return self.layersDict[i](x)
        
        lambdaList = []
        for i in range(len(thetaDict)):
            #lambdaList.append(lambda x : self.layersDict[i](x))
            lambdaList.append(findLayer(i, x))
        
        return tf.convert_to_tensor(tf.experimental.numpy.select(condList, lambdaList))

    
        
        
#def findLayer(i, x):
            #return self.layersDict[i](x)
        
        #self.lambdaList = []
        #for i in range(len(thetaDict)):
         #   layer = findLayer(i, x)
            #lambdaList.append(lambda x : self.layersDict[i](x))
         #   self.lambdaList.append(layer)
        
        #return tf.convert_to_tensor(tf.experimental.numpy.select(condList, self.lambdaList))
        
        
model = model_1()
print("Step 5 done.")

#######  LOAD WEIGHTS ####################

#model.load_weights('./trained_weights/trained_weights_(<6)_(1024withGAUSS)_v0')

##########################################

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

print("Step 6 done.")
EPOCHS = 1000
steps_per_epoch = 5
#model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)

model.summary()

#Saves model
model.save_weights('./trained_weights/trained_weights_NEW_v0')
#model.save('./trained_models/trained_model_previous_v0')
