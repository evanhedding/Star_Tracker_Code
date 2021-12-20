import numpy as np
import csv
import pickle

# min/max magnitudes for guidestars. 3-6 gives ~6600, 2-7 ~21000
MIN_MAG = 3
MAX_MAG = 6

# Radius of donut to build triangles in. Wider range gives MANY more triangles
MIN_DEG = 5
MAX_DEG = 6
MIN_DEG_DIST = np.sin(np.radians(MIN_DEG))/np.sin(np.radians((180-MIN_DEG)/2))
MAX_DEG_DIST = np.sin(np.radians(MAX_DEG))/np.sin(np.radians((180-MAX_DEG)/2))

#stars_or: [source_id, asc, dec, mag]
stars_og = np.genfromtxt('Star Tracker (< 7.5 mag).csv',delimiter=',')

# *Remember, first row is nan
#np.array([source id, mag, ux, uy, uz])
stars_array = np.zeros((stars_og.shape[0]-1, 5))
for star in range(1, stars_og.shape[0]-1):
    curr_id = stars_og[star, 0]
    curr_asc = np.radians(stars_og[star, 1])
    curr_dec = np.radians(stars_og[star, 2])
    curr_mag = stars_og[star, 3]
    stars_array[star,:] = np.array([curr_id, curr_mag, np.cos(curr_asc)*np.cos(curr_dec), np.sin(curr_asc)*np.cos(curr_dec), np.sin(curr_dec)])
    
########################################################################
num_stars = stars_array.shape[0]

min_neighbors = 1000
max_neighbors = 0

guidestarInfo_dict = {}
guidestarNeighbors_dict = {}
for i in range(num_stars):
    if i % 10 == 0:
        print(i*100/num_stars, " %")
    
    # Only use stars where 7 > mag > 2 for guidestars?
    if stars_array[i, 1] >= MIN_MAG and stars_array[i, 1] <= MAX_MAG:
        curr_guidestar_vec = stars_array[i, 2:]
        curr_guidestar_id = stars_array[i, 0]
        
        # List of stars in range
        neighbor_stars = []
        for j in range(num_stars):
            curr_neighbor_vec = stars_array[j, 2:]
            dist = np.linalg.norm(curr_guidestar_vec - curr_neighbor_vec)
            if dist >= MIN_DEG_DIST and dist <= MAX_DEG_DIST:
                neighbor_stars.append(stars_array[j,:])
                
        if len(neighbor_stars) < min_neighbors:
            min_neighbors = len(neighbor_stars)
            print("Min Neighs: ", min_neighbors)
            
        if len(neighbor_stars) > max_neighbors:
            max_neighbors = len(neighbor_stars)
            print("Max Neighs: ", max_neighbors)
        
        
        guidestarInfo_dict[curr_guidestar_id] = stars_array[i,:]
        guidestarNeighbors_dict[curr_guidestar_id] = neighbor_stars

print("Final Min: ", min_neighbors)
print("Final Max: ", max_neighbors)

with open("guidestarInfo_dict_36", "wb") as fp:
    pickle.dump(guidestarInfo_dict, fp)
    
with open("guidestarNeighbors_dict_36", "wb") as fp:
    pickle.dump(guidestarNeighbors_dict, fp)


