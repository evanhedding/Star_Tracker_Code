import numpy as np
import csv
import pickle
import tensorflow as tf

with open("./test_samples/test_data.txt", "rb") as fp:
    test_data = pickle.load(fp)

with open("./test_samples/test_labels.txt", "rb") as fp:
    test_labels = pickle.load(fp)
    
with open("./test_samples/labels_map.txt", "rb") as fp:
    labels_map = pickle.load(fp)
    
num_stars = len(test_labels)

class model_1(tf.keras.Model):
    def __init__(self):
        super(model_1, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.05)
        
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
        
        x = self.d2(x)
        x = self.drop(x)
        
        x = self.d3(x)
        x = self.n3(x)
        
        x = self.d4(x)
        x = self.drop(x)
        
        x = self.d5(x)
        x = self.n5(x)
        
        x = self.d6(x)
        x = self.drop(x)
        
        x = self.d7(x)
        x = self.n7(x)
        
        return self.d8(x)
        
model = model_1()
model.load_weights('./trained_weights/trained_weights_(<6)_(1024withGAUSS)_v0')
model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

n = 0
wins = 0
for i in range(len(test_data)):
    identifier_matrix = tf.convert_to_tensor(test_data[i])
    identifier_matrix = tf.reshape(identifier_matrix, (1,1024))
    #print(identifier_matrix)
    #print(tf.argmax(model.predict(identifier_matrix)[0], axis=0).numpy())
    
    #print(test_labels[i])
    index = tf.argmax(model.predict(identifier_matrix)[0], axis=0).numpy()
    pred_star = labels_map[index]
    #print(pred_star)
    n += 1
    if pred_star == test_labels[i]:
        wins += 1
        print("Win!! Current Accuracy: ", wins*100/n, "%")
    
        
print("Accuracy Total = ", wins*100/num_stars, "%")

