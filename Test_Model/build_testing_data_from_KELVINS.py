import numpy as np
import csv
import pickle
import tensorflow as tf

with open("./test_samples/test_labels.txt", "rb") as fp:
    test_labels = pickle.load(fp)
    
with open("./test_samples/labels_map.txt", "rb") as fp:
    labels_map = pickle.load(fp)
    
num_stars = 4024 #len(test_labels)

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
        self.d5 = tf.keras.layers.Dense(4096) #**Without these we got to 95% at epoch 9.
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
model.load_weights('./trained_weights/trained_weights_(<6)_(1024withNoise)_v3')
model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])



FOV_DEG = 10
FOV_DIST = np.sin(np.radians(FOV_DEG))/np.sin(np.radians((180-FOV_DEG)/2))

e1 = 0.03
e2 = 0.17
scale = 80
INPUT_SIZE = 1024
INPUT_NUM_STARS = 10
def buildModelInput(scene, run):
    stars_to_check = []
    print("Run :", run+1)
    print()
    
    size = len(scene)
    if len(scene) > INPUT_NUM_STARS:
        size = INPUT_NUM_STARS
        
    for k in range(len(scene)):
        curr_star_dists = np.zeros((len(scene), 4))
        curr_guidestar_vec = scene[k]
        
        for j in range(len(scene)): #Check distance to every star in the window
            dist = np.linalg.norm(curr_guidestar_vec - scene[j])
            if dist < FOV_DIST:
                curr_star_dists[j, 0] = dist*scale
                curr_star_dists[j, 1:] = scene[j]
            
        
        curr_star_dists = curr_star_dists[np.argsort(curr_star_dists[:,0])]
        curr_star_dists = curr_star_dists[:size,:]
        surrounding_star_dists = np.zeros(size*size)
        guide_star_dists = curr_star_dists[:,0]
        
        for m in range(size):
            for n in range(size):
                surrounding_star_dists[m*size + n] = np.linalg.norm(curr_star_dists[m, 1:] - curr_star_dists[m-n, 1:])*scale
        
       
        max1 = int(np.ceil(guide_star_dists[np.argmax(guide_star_dists)]/e1))
        max2 = int(np.ceil(surrounding_star_dists[np.argmax(surrounding_star_dists)]/e2))
        D1 = np.zeros(max1+4)
        D2 = np.zeros(max2+4)
        
        #print(max1, max2)
        
        for x in range(2, max1):
            for y in range(size):
                if int(np.ceil(guide_star_dists[y]/e1)) == x:
                    D1[x] = D1[x] + 9
                    D1[x-1] = D1[x-1] + 3
                    D1[x+1] = D1[x+1] + 3
                    D1[x-2] = D1[x-2] + 1
                    D1[x+2] = D1[x+2] + 1
                    
        for x in range(2, max2):
            for y in range(size*size):
                if int(np.ceil(surrounding_star_dists[y]/e2)) == x:
                    D2[x] = D2[x] + 4
                    D2[x-1] = D2[x-1] + 2
                    D2[x+1] = D2[x+1] + 2
                    D2[x-2] = D2[x-2] + 1
                    D2[x+2] = D2[x+2] + 1
                    
        #print(D1, D2)
        input = np.zeros(INPUT_SIZE)
        features = np.append(D1/np.linalg.norm(D1), D2/np.linalg.norm(D2))
        input[:len(features)] = features
        stars_to_check.append(input)
        
        
    return stars_to_check


scenes = []
k=0
with open('input_sample.csv', 'r') as csvfile:
    input_csv = csv.reader(csvfile)
    for row in input_csv:
        input_array = np.genfromtxt(row)
        curr_scene = []
        for i in range(int(len(input_array)/3)):
            #print(i)
            curr_star = np.zeros(3)
            curr_star[0] = input_array[i*3]
            curr_star[1] = input_array[i*3 + 1]
            curr_star[2] = input_array[i*3 + 2]
            curr_scene.append(curr_star)
        #print("input array", input_array)
        #print("curr scene", curr_scene)
        scenes.append(curr_scene)


test_samples = []
for i in range(len(scenes)):
    stars_to_check = buildModelInput((scenes[i]), i)
    test_samples.append(stars_to_check)
    for star in stars_to_check:
        identifier_matrix = tf.convert_to_tensor(star)
        identifier_matrix = tf.reshape(identifier_matrix, (1,1024))
        odds = model.predict(identifier_matrix)[0]
        index = tf.argmax(odds, axis=0).numpy()
        pred_star = labels_map[index]
        print(pred_star, odds[index])


#with open("Kelvins_testing_samples.txt", "wb") as fp:
#   pickle.dump(test_samples, fp)



#for i in range(10):
#    print("Run number", i)
#   print()
#    for star in test_samples[i]:
#        identifier_matrix = tf.convert_to_tensor(star)
#        identifier_matrix = tf.reshape(identifier_matrix, (1,512))
#        odds = model.predict(identifier_matrix)[0]
#        index = tf.argmax(odds, axis=0).numpy()
#        pred_star = labels_map[index]
#        print(pred_star, odds[index])
#    print()


