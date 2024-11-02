import numpy as np

def function(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    
    y1 = x1 * x3 + 1.2 * x1 * x5 - x6 * x7 * x8 - 2 * x1**2 * x8 + x5
    y2 = x1 * x5 * x6 - x3 * x4 - 3 * x2 * x3 + 2 * x2**2 * x4 - 2 * x7 * x8 - 1
    y3 = 2 * x3**2 - x5 * x7 - 3 * x1 * x4 * x6 - x1**2 * x2 * x4 - 1
    y4 = -x6**3 + 2 * x1 * x3 * x8 - x1 * x4 * x7 - 2 * x5**2 * x2 * x4 - x8
    y5 = x1**2 * x5 - 3 * x3 * x4 * x8 + x1 * x2 * x4 - 3 * x6 - x1**2 * x7 + 2
    y6 = x1**2 * x3 * x6 - x3 * x5 * x7 + x3 * x4 + 2.2 * x4 + x2**2 * x3 - 2.1
    
    return [y1, y2, y3, y4, y5, y6]

Nt = 1000 # Training Data
Nv = 200  # Validation Data
noiseMean = 0.0
noiseStddev = 0.001


trainingData = []
loop = 0

while loop < Nt:
    x = np.random.random(8).tolist()
    y = function(x)

    for n in range(len(x)): x[n] = round(x[n], 2)
    for n in range(len(y)): y[n] = round(y[n], 2)
    
    trainingData.append([x, y]) 

    print("x", loop, " : ", x)        
    print("y", loop, " : ", y)
    
    loop += 1

print(trainingData)