import numpy as np

x = np.zeros(1024)

for i in range(1024):
    x[i] = np.sin(2.7*i/100)
    
print(x)
