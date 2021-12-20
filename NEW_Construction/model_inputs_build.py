import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt


with open("./dictionaries_36/guidestarDict_with_features", "rb") as fp:
    guidestarDict_with_features = pickle.load(fp)
    
with open("./dictionaries_36/id2label", "rb") as fp:
    id2label_dict = pickle.load(fp)
    
maxList = []
minList = []
maxNum = 0
minNum = 10000

for guidestar_id in guidestarDict_with_features:
    if len(guidestarDict_with_features[guidestar_id]) > maxNum:
        maxNum = len(guidestarDict_with_features[guidestar_id])
        maxList = guidestarDict_with_features[guidestar_id]
    if len(guidestarDict_with_features[guidestar_id]) < minNum:
        minNum = len(guidestarDict_with_features[guidestar_id])
        minList = guidestarDict_with_features[guidestar_id]
        
print(maxNum)
print(minNum)

#plt.figure(1)
#plt.plot(np.sort(np.asarray(maxList), 0)[:,0])
#plt.show()

scatteredFeatures = []
rangesList = []

minVec = np.sort(np.asarray(minList)[:,0])
for i in range(len(maxList)):
    



#for i in range(maxNum):
    





#with open("guidestarDict_with_features_36", "wb") as fp:
#    pickle.dump(guidestarDict_with_features, fp)


