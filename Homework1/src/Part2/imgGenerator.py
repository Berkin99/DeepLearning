
import sys
import os
import subprocess
import random

generatorPath = 'C:/Users/berki/Desktop/YeniaySrc/WorkspaceAI/DeepLearning/Homework1/src/Part2/2D-Shape-Generator/main.py'

def generate(inx:int, shape:str, pixels:int, movex:int, movey:int, rot:float, scalex:float, scaley:float):
    
    args = [
        'python', generatorPath,
        '--name', str(inx),
        '--shapes', shape,
        '--canvas_size', str(pixels), str(pixels),
        '--stim_trx', str(movex),
        '--stim_try', str(movey),
        '--stim_rota', str(rot),
        '--stim_scale', str(scalex), str(scaley)
    ]
    subprocess.run(args)

def generateRandom(inx:int, shape:str,  pixels = 128, posSigma = 20, maxscale = 2.5, scaleSigma=0.2):

    mx = int(random.gauss(pixels / 2, posSigma))
    my = int(random.gauss(pixels / 2, posSigma))
    mx = min(max(mx, 0), pixels)
    my = min(max(my, 0), pixels)
    
    mscl = random.uniform(0.5,maxscale)
    scx = random.gauss(mscl, scaleSigma)
    scy = random.gauss(mscl, scaleSigma)
    scx = max(scx, 0)
    scy = max(scy, 0)
    
    generate(
        inx=inx, 
        shape=shape, 
        pixels=pixels,
        movex=mx, movey=my,
        rot=random.uniform(0, 360),
        scalex=scx, scaley=scy 
        )


def main():

    args = sys.argv[1:]  

    if len(args) < 2:
        print("Not enough arguments") 
        return
    
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir): os.makedirs(output_dir)    
    
    if len(args) == 3:
        for i in range(int(args[0]),int(args[1])):
            generateRandom(i, args[2])
    else: generateRandom(int(args[0]), args[1])

if __name__ == '__main__':
    main()