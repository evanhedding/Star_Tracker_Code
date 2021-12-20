import numpy as np
import csv
import pickle
from rotation import *

e1 = 0.03
e2 = 0.17
scale = 120
INPUT_SIZE = 1024
INPUT_NUM_STARS = 10
with open("GRID_scenes_10FOV_2deg_7.5mag_____(withRADIANS).txt", "rb") as fp:
    scenes = pickle.load(fp)

print("num scenes", len(scenes))

min = 10000
max = 0
for i in range(len(scenes)):
    stars = len(scenes[i])
    if stars < min:
        min = stars
    if stars > max:
        max = stars
print("min", min)
print("max", max)

max = 0
star_dict = {}
for i in range(len(scenes)):
    print(i*100/len(scenes), " %")
   
    for k in range(0, INPUT_NUM_STARS): #Top ten(?) stars within the current window
        curr_star_dists = np.zeros((len(scenes[i]), 4)) #Stores dist and mag from current top star
        curr_guidestar_vec = scenes[i][k][2:] #Unit vector of current top-ten star
        curr_guidestar_id = scenes[i][k][0] #Source_id, This will be the key for this guidestar
        
        num_stars_to_add = 0
        if len(scenes[i]) < INPUT_NUM_STARS: num_stars_to_add = len(scenes[i])
        else: num_stars_to_add = INPUT_NUM_STARS
        
        addNoise = False
        
        for j in range(len(scenes[i])): #Check distance to every star in the window
            curr_star = scenes[i][j][2:]
            if addNoise:
                # Star centroiding noise (position) estimated to be in the range of +-0.0003 deg
                # I model this by rotating the measured star vector 3 times across each axis
                # with a randomly selected level of noise in the range mentioned above.
                
                noise_rotx = np.random.uniform(-8e-4, 8e-4)
                noise_roty = np.random.uniform(-8e-4, 8e-4)
                noise_rotz = np.random.uniform(-8e-4, 8e-4)
                curr_star = rotx(noise_rotx)@roty(noise_roty)@rotz(noise_rotz)@curr_star
                
            dist = np.linalg.norm(curr_guidestar_vec - curr_star)
            curr_star_dists[j, 0] = dist*scale
            curr_star_dists[j, 1:] = scenes[i][j][2:]
         
        curr_star_dists = curr_star_dists[np.argsort(curr_star_dists[:,0])]
        curr_star_dists = curr_star_dists[:num_stars_to_add,:]
        surrounding_star_dists = np.zeros(num_stars_to_add*num_stars_to_add)
        guide_star_dists = curr_star_dists[:,0]
        
        for m in range(num_stars_to_add):
            for n in range(num_stars_to_add):
                surrounding_star_dists[m*num_stars_to_add + n] = np.linalg.norm(curr_star_dists[m, 1:] - curr_star_dists[m-n, 1:])*scale
            
            
        #print(guide_star_dists)
        max1 = int(np.ceil(guide_star_dists[np.argmax(guide_star_dists)]/e1))
        max2 = int(np.ceil(surrounding_star_dists[np.argmax(surrounding_star_dists)]/e2))
        #print(max1, max2)
        D1 = np.zeros(max1+4)
        D2 = np.zeros(max2+4)
        
        #print(features)
        
        for x in range(2, max1):
            for y in range(num_stars_to_add):
                if int(np.ceil(guide_star_dists[y]/e1)) == x:
                    D1[x] = D1[x] + 9
                    D1[x-1] = D1[x-1] + 3
                    D1[x+1] = D1[x+1] + 3
                    D1[x-2] = D1[x-2] + 1
                    D1[x+2] = D1[x+2] + 1
                    
        for x in range(2, max2):
            for y in range(num_stars_to_add*num_stars_to_add):
                if int(np.ceil(surrounding_star_dists[y]/e2)) == x:
                    D2[x] = D2[x] + 4
                    D2[x-1] = D2[x-1] + 2
                    D2[x+1] = D2[x+1] + 2
                    D2[x-2] = D2[x-2] + 1
                    D2[x+2] = D2[x+2] + 1
        
        input = np.zeros(INPUT_SIZE)
        features = np.append(D1/np.linalg.norm(D1), D2/np.linalg.norm(D2))
        input[:len(features)] = features
        
        #print(input)
        if len(features) > max:
            max = len(features)
            
        if curr_guidestar_id in star_dict:
            star_dict[curr_guidestar_id].append(input)
        else:
            star_dict[curr_guidestar_id] = []
            star_dict[curr_guidestar_id].append(input)

print("max: ", max)
print("num stars in dict", len(star_dict))


with open("training_data_with_GAUSS_10FOV_2deg_7mag_(512)_v0.txt", "wb") as fp:
    pickle.dump(star_dict, fp)

