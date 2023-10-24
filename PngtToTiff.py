import cv2
from pathlib import Path
import imageio
import os
def pngToTiff(path, index, name):
    print("Started tiffing "+str(index))
    imageDict = {}
    for j in list(((Path(path).glob("*.png")))):
        #print(j)
        num = int(str(os.path.split(os.path.splitext(j)[0])[1]))
        #print(num)
        image = cv2.imread(str(j))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
        imageDict[num] = image

    tiffArray = []
    for i in sorted(imageDict):
        #print(i)
        tiffArray.append(imageDict.get(i))
    imageio.mimwrite("Outputs/"+name+".tiff",tiffArray)
    print("Done tiffing "+str(index))
    #for i in imageArray:
        #print(i[0])