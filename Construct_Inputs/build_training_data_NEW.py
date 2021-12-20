import numpy as np
import csv
import pickle
from rotation import *

NUM_GUIDESTARS = 15
MAX_INPUT_STARS = 10

WAVELENGTH = 1024
K_FREQ = 10
K_AMP = 100

def makeFeature(array):
    array = array/np.linalg.norm(array)
    # Takes in an array of stars, sized (n,3).
    # The first star is the guidestar, the rest are surrounding stars.
    guidestar = array[0,:]
    surrounding_stars = array[1:,:]
    #avg_dists = np.zeros(array.shape[0])
    
    tot_dist = 0
    for i in range(surrounding_stars.shape[0]):
        tot_dist = tot_dist + np.linalg.norm(guidestar - surrounding_stars[i,:])
    
    # Average distance from all surrounding stars to the guidestar.
    # To be used as freq and amp multitude for waveform.
    freq = tot_dist/float(surrounding_stars.shape[0])
    
    # Now get avg distances from each surrounding star to each other surrounding star.
    #for i in range(surrounding_stars.shape[0]):
    #    curr_tot_dist = 0
    #    for j in range(surrounding_stars.shape[0]):
    #        if i != j:
    #            curr_tot_dist = curr_tot_dist + np.linalg.norm(surrounding_stars[i,:] - surrounding_stars[i-j,:])
    #    avg_dists[i+1] = curr_tot_dist/(surrounding_stars.shape[0] - 1)

    #avg_dists = avg_dists/np.linalg.norm(avg_dists)
    
    #freq = avg_dists[0]
    #amplitudes = avg_dists[1:]
    
    #feature = np.zeros((MAX_INPUT_STARS - 1, WAVELENGTH))
    feature = np.zeros(WAVELENGTH)
    #for i in range(len(amplitudes)):
    for t in range(WAVELENGTH):
        feature[t] = K_AMP*np.sin(freq*t/K_FREQ)
        
    return feature # Sized (number_surrounding_stars, WAVELENGTH)
    


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
        
        #for n in range(3, MAX_INPUT_STARS):
        input = makeFeature(curr_star_dists[:MAX_INPUT_STARS])
        
        #print(input)
        
        if curr_guidestar_id in star_dict:
            star_dict[curr_guidestar_id].append(input/np.linalg.norm(input))
        else:
            star_dict[curr_guidestar_id] = []
            star_dict[curr_guidestar_id].append(input/np.linalg.norm(input))
        
        
print("num stars in dict", len(star_dict))


with open("training_data_BRANDNEW_10FOV_2deg_7.5mag_____(withRADIANS)_v2.txt", "wb") as fp:
    pickle.dump(star_dict, fp)

