import numpy as np
import csv
import pickle
import tensorflow as tf

MIN_SAMPLES_PER_STAR = 50
MAX_SAMPLES_PER_STAR = 100

with open("training_data_BRANDNEW_10FOV_2deg_7.5mag_____(withRADIANS)_v2.txt", "rb") as fp:
    star_dict = pickle.load(fp)
    
def MinMax():
    min=10000
    max=0
    for star in star_dict:
        if len(star_dict[star]) < min:
            min = len(star_dict[star])
        if len(star_dict[star]) > max:
            max = len(star_dict[star])
    print("min", min)
    print("max", max)

MinMax()
####### Removes any guidestar with <=5 features scenes to train on #####

print("total stars in star_dict before: ", len(star_dict))
stars_to_delete = []
for star in star_dict:
    if len(star_dict[star]) < MIN_SAMPLES_PER_STAR:
        stars_to_delete.append(star)
        
    while len(star_dict[star]) > MAX_SAMPLES_PER_STAR:
        rand_sample = np.random.randint(0, len(star_dict[star]))
        star_dict[star].pop(rand_sample)
        
for star in stars_to_delete:
    star_dict.pop(star)

print("total stars in star_dict after: ", len(star_dict))

MinMax()
########################################################################

print("Step 1 done.")
num_stars = len(star_dict)
train_data = []
test_data = []
labels = []
test_labels = []
val_labels = []
val_data = []
labels_map = np.zeros(num_stars) #Maps one-hot indices to star number in dictionary
n = 0
for star in star_dict:
    label = np.zeros(num_stars) #Converting labels to one-hot
    label[n] = 1 #One-hot representation
    #print(n, label)
    labels_map[n] = star
    
    #Build test data
    #print("before", (len(star_dict[star])))
    idx = np.random.randint(0, len(star_dict[star]))
    test_data.append(star_dict[star].pop(idx))
    test_labels.append(star)
    
    for k in range(15):
        idx = np.random.randint(0, len(star_dict[star]))
        val_data.append(star_dict[star].pop(idx))
        val_labels.append(label)
    
    for i in range(len(star_dict[star])):
        train_data.append(star_dict[star][i])
        labels.append(label)
        
    n += 1  #new
    #print("after", len(star_dict[star]))
    
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
    
with open("./test_samples/labels_map.txt", "wb") as fp:
    pickle.dump(labels_map, fp)

print("Step 3 done.")
val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).shuffle(10000).batch(100)
print("Step 4.1 done.")
train_ds = tf.data.Dataset.from_tensor_slices((train_data, labels)).shuffle(10000).batch(500).repeat()
print("Step 4.2 done.")

########### NEW IDEAS #############################

# You need to branch the layers or train different models and save the weights.

# Option 1: Train one model to match a star pattern to a scence, aka 10 deg FOV.
#           - Then train a simple model to recognize maybe 6? groups of scenes.
#           - Need to completely redo the training data for this with new labels for each group.
#               - I need new datasets completely. OR do I?!!? Can I just use the same full star dataset as input but now just output a group number (i.e. output size of 6)?
#               - Train that model to recognize a group number and output that in one-hot format. (i.e [0,0,0,0,0.99,0,0])

#          *** All I need to do this with minimal code changes if I can get a group number into each star in the star_dict.
#               - Just add the group number to the beginning of list for each star

#           - Then I can load that model into the first few layers of the FULL model and make those layers untrainable.
#           - Create argmax of the outpur of those layers and then move to new layers depending on that index value!

#

# Option 2: Train only one model, but branch the top layers to focus on a specific scene instead of one model fully connected model for the entire night's sky.
#           - This would need altering of the training data I believe, in order to be able to differentiate between scenes during the call() function. (Not entirely sure how to do this)

###################################################

class model_1(tf.keras.Model):
    def __init__(self):
        super(model_1, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.05)
        #self.drop = tf.keras.layers.GaussianDropout(0.05)
        
        self.n1 = tf.keras.layers.BatchNormalization()
        self.n3 = tf.keras.layers.BatchNormalization()
        self.n5 = tf.keras.layers.BatchNormalization()
        self.n7 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(1024)
        self.d2 = tf.keras.layers.Dense(1024, activation='tanh')
        self.d3 = tf.keras.layers.Dense(2048)
        self.d4 = tf.keras.layers.Dense(2048, activation='tanh')
        self.d5 = tf.keras.layers.Dense(4096)
        self.d6 = tf.keras.layers.Dense(4096, activation='tanh')
        self.d7 = tf.keras.layers.Dense(num_stars)
        self.d8 = tf.keras.layers.Dense(num_stars, activation='softmax')
        
        ###############################################
        
    def call(self, x):
        x = self.d1(x)
        x = self.n1(x)
        #x = self.drop(x)
        
        x = self.d2(x)
        x = self.drop(x)
        
        x = self.d3(x)
        x = self.n3(x)
        #x = self.drop(x)
        
        x = self.d4(x)
        x = self.drop(x)
        
        x = self.d5(x)
        x = self.n5(x)
        #x = self.drop(x)
        
        x = self.d6(x)
        x = self.drop(x)
        
        x = self.d7(x)
        x = self.n7(x)
        #x = self.drop(x)
        
        return self.d8(x)
        
        
model = model_1()
print("Step 5 done.")

#######  LOAD WEIGHTS ####################

#model.load_weights('./trained_weights/trained_weights_(<6)_(1024withGAUSS)_v0')

##########################################

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

print("Step 6 done.")
EPOCHS = 10
steps_per_epoch = 500
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)

model.summary()

#Saves model
model.save_weights('./trained_weights/trained_weights_NEW_v0')
#model.save('./trained_models/trained_model_previous_v0')
