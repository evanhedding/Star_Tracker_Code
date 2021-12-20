import numpy as np
import csv
import pickle
import rotation as r

FOV_DEG = 10
FOV_DIST = np.sin(np.radians(FOV_DEG))/np.sin(np.radians((180-FOV_DEG)/2))

# What I want to do...
# 1. Create a unit vector for each star given asc/dec
# 2. Decide on field of view square for camera
# 3. Create training data
#    Options: - Choose center star and do bin/histogram method using distances from that center star.
#             - Or some other way of creating images, ensuring padding and overlapping.
#             - Or maybe making an algorithm to systematically group clusters and train on that?
#
#             - Break spherical view frame into sections based on camera FOV
#                   - Reshape matrix into a 3D numpy array
#                       0 - FOV section
#                       1 - star identifier
#                       2 - star unit vector followed by magnitude

#stars_or: [source_id, asc, dec, mag]
stars_og = np.genfromtxt('Star Tracker (< 7 mag).csv',delimiter=',')

# *Remember, first row is nan

##################### NEW IDEA #################################
#   Take circular scenes, 10 deg rotation.
#   - Start with [1,0,0] as center star
#   - Go through each star, check if within dist of center star
#           - dist from trig = sin(chosen FOV deg)/sin((180-deg)/2)
#                   - For 10 deg, we have dist = sin(10)/sin(85)
#   - Rotate the center star and restart

#np.array([source id, mag, ux, uy, uz])
stars_array = np.zeros((stars_og.shape[0]-1, 5))
for star in range(1, stars_og.shape[0]-1):
    curr_id = stars_og[star, 0]
    curr_asc = stars_og[star, 1]
    curr_dec = stars_og[star, 2]
    curr_mag = stars_og[star, 3]
    stars_array[star,:] = np.array([curr_id, curr_mag, np.cos(curr_asc)*np.cos(curr_dec), np.sin(curr_asc)*np.cos(curr_dec), np.sin(curr_dec)])
    
#Fill this list with scenes based on NEW IDEA above.
scenes = []
for rot_y in range(-90,90,2):
    print(rot_y)
    print("")
    
    og = np.array([1,0,0])
    center_star_y = r.roty(rot_y)@og
    for rot_z in range(0,360,2):
        #print(rot_z)
        center_star = r.rotz(rot_z)@center_star_y
        curr_scene = []
        for i in range(stars_array.shape[0]):
            dist = np.linalg.norm(center_star - stars_array[i, 2:])
            if dist <= FOV_DIST/2: # Diameter in next step is FOV_DIST, therefore need half that here
                curr_scene.append(stars_array[i,:])
        #if len(curr_scene) > 10:
        scenes.append(curr_scene)
            

min = 10000
for i in range(len(scenes)):
    stars = len(scenes[i])
    print("num stars", stars)
    if stars < min:
        min = stars
print("min", min)
    
with open("GRID_scenes_10FOV_2deg_7mag.txt", "wb") as fp:
    pickle.dump(scenes, fp)

#Then, in each scene, choose the 3 brightest stars and spin around those?
