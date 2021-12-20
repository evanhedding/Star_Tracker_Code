import numpy as np
import np.linalg.norm as norm
import csv
import pickle
from rotation import *

NUM_GUIDESTARS = 15
MAX_INPUT_STARS = 10
NUM_TRIANGLES = 10

FEATURE_LENGTH = 6

def makeFeature(starsArr):
    # Takes in an array of three stars, sized (3,3).
    # The first star is the guidestar, the rest are surrounding stars.
    
    # i.e. AB denotes the vector from A to B, or B - A in vector subtraction
    AB = starsArr[1,:] - starsArr[0,:]
    AC = starsArr[2,:] - starsArr[0,:]
    BA = starsArr[0,:] - starsArr[1,:]
    BC = starsArr[2,:] - starsArr[1,:]
    CB = starsArr[1,:] - starsArr[2,:]
    CA = starsArr[0,:] - starsArr[2,:]
    
    thA = np.arccos(np.dot(AB, AC)/(norm(AB)*norm(AC)))
    thB = np.arccos(np.dot(BA, BC)/(norm(BA)*norm(BC)))
    thC = np.arccos(np.dot(CB, CA)/(norm(CB)*norm(CA)))
    
    d_norm = np.sqrt(norm(AB)**2 + norm(BC)**2 + norm(CA)**2)
        
    return np.array([thA, thB, thC, norm(AB)/d_norm, norm(BC)/d_norm, norm(CA)/d_norm])
    


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


star_dict = {}
for i in range(len(scenes)):
    if i % 10 == 0:
        print(i*100/len(scenes), " %")
        ############################################
        # Implement an estimated time remaining function. This shit takes a while.
        ############################################
   
    for k in range(NUM_GUIDESTARS): #Top NUM_GUIDESTARS magnitude stars within the current window. (Scenes come ordered by mag)
        
        curr_star_dists = np.zeros((len(scenes[i]), 4))
        curr_guidestar_vec = scenes[i][k][2:] #Unit vector of current top-ten star
        curr_guidestar_id = scenes[i][k][0] #Source_id, This will be the key for this guidestar
        
        for j in range(len(scenes[i])): #Check distance to every star in the window
            curr_star = scenes[i][j][2:]
            ########################################################################
            #addNoise = False
            #if addNoise:
            #    # Optional noise on each star's unit vector
            #   noise_rotx = np.random.uniform(-8e-4, 8e-4)
            #    noise_roty = np.random.uniform(-8e-4, 8e-4)
            #    noise_rotz = np.random.uniform(-8e-4, 8e-4)
            #    curr_star = rotx(noise_rotx)@roty(noise_roty)@rotz(noise_rotz)@curr_star
            ########################################################################
            dist = np.linalg.norm(curr_guidestar_vec - curr_star)
            curr_star_dists[j, 0] = dist
            curr_star_dists[j, 1:] = scenes[i][j][2:]
         
        curr_star_dists = curr_star_dists[np.argsort(curr_star_dists[:,0])]
        curr_star_dists = curr_star_dists[:MAX_INPUT_STARS,:]
        
        for n in range(3, MAX_INPUT_STARS):
            input = makeFeature(curr_star_dists[:MAX_INPUT_STARS])
            
            #print(input)
            
            if curr_guidestar_id in star_dict:
                star_dict[curr_guidestar_id].append(input)
            else:
                star_dict[curr_guidestar_id] = []
                star_dict[curr_guidestar_id].append(input)
        
        
print("num stars in dict", len(star_dict))


with open("training_data_BRANDNEW_10FOV_2deg_7.5mag_____(withRADIANS)_v2.txt", "wb") as fp:
    pickle.dump(star_dict, fp)

