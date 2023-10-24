import cv2
import numpy as np 
import os
import shutil
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

st = time.time()
tf.get_logger().setLevel('ERROR')
from pathlib import Path
from PIL import Image, ImageSequence
import ThresholdContour
import PngtToTiff
import pickle

from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential

# path to input image is specified and  
# image is loaded with imread command 
  
thresholdInt = 40
blurInt = 2
contourX = 20
contourY = 20

EardrumCenters = [[] for _ in range(417)] 
#print(len(EardrumCenters))
totalContours = [[] for _ in range(417)]
img_height = 200
img_width = 80


print("Started")

accuracyScore = 0




def makeDelDirs(index):
    print("Making and Deleting Directories")

    
    try:
        os.mkdir("Inputs")
    except:
        pass
    try:
        os.mkdir('Images')
    except:
        shutil.rmtree("Images")
        os.mkdir('Images')
        pass

    try:
        os.mkdir('Outputs')
    except:
        pass
   
    

    


    
    os.mkdir("Images/CroppedImages("+str(index)+")")
    os.mkdir("Images/EdgeDetectedCropped("+str(index)+")")
    os.mkdir("Images/EdgeDetectedCropped("+str(index)+")2")
    os.mkdir("Images/SortedImages("+str(index)+")")
    os.mkdir("Images/SortedImages("+str(index)+")2")
    os.mkdir("Images/SortedImages("+str(index)+")/Eardrum/")
    os.mkdir("Images/SortedImages("+str(index)+")/CroppedImages/")
    os.mkdir("Images/SortedImages("+str(index)+")/Interference/")
    os.mkdir("Images/SortedImages("+str(index)+")/BoundedImages/")
    os.mkdir("Images/SortedImages("+str(index)+")/MaskedImages2/")
    os.mkdir("Images/SortedImages("+str(index)+")2/MaskedImages2/")
    os.mkdir("Images/SortedImages("+str(index)+")2/Eardrum/")
    os.mkdir("Images/SortedImages("+str(index)+")2/Interference/")
    os.mkdir("Images/SortedImages("+str(index)+")/AvgImages/")
    os.mkdir("Images/BoxTests")
    os.mkdir("Images/BoxTests2("+str(index)+")")
    os.mkdir("Images/MaskedImages("+str(index)+")")
    os.mkdir("Images/MaskedImages("+str(index)+")2/")
    os.mkdir("Images/Masks("+str(index)+")2/")
    os.mkdir("Images/Masks("+str(index)+")2/Interference/")
    os.mkdir("Images/Masks("+str(index)+")2/Eardrum/")
    os.mkdir("Images/Masks("+str(index)+")2/Crops/")


def runMain(index):
    global accuracyScore
        
    eardrumArrays = [[] for _ in range(417)]

    
    #print("Inputs: "+ str(inputs))
    #print(list(inputs.glob('*.png')))
    for i in list(inputs.glob('*.png')):
        Objects = []
        contourList = []
        #print((eardrumArrays))
        inputIndex = int(str(os.path.split(os.path.splitext(i)[0])[1]).replace('Layer',''))-1
        #print(inputIndex)
        try:
            os.mkdir("Images/EdgeDetectedCropped("+str(index)+")/"+str(os.path.split(os.path.splitext(i)[0])[1]))
        except:
            pass
        
        filePath = str(i)

        image1 = cv2.imread(filePath) 
        image2 = image1.copy()
        
        counter = 0
        image1Gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY )

        ret, image1Thresh = cv2.threshold(image1Gray, thresholdInt, 255, cv2.THRESH_BINARY)
        blurred = cv2.blur(image1Thresh.copy(), (blurInt,blurInt))

        contours, _ = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cropped_image = image1
        
        maskImage = image1.copy()
        interferenceMask = image1.copy()
        counter = -1
        contourX =30
        contourY=30
        for c in contours:
            counter+=1
            rect = cv2.boundingRect(c)
            if rect[2] < contourX or rect[3] < contourY: continue
            x,y,w,h = rect
            
            cropped_image = image1Thresh[y:y+h, x:x+w]
            cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
            image1GrayArray = cropped_image
            img_tf = tf.convert_to_tensor(image1GrayArray)
            Objects.append(img_tf)
            contourList.append(c)
        cv2.imwrite("Images/MaskedImages("+str(index)+")/"+os.path.basename(os.path.splitext(filePath)[0])+".png",maskImage)
        cv2.imwrite("Images/MaskedImages("+str(index)+")2/"+os.path.basename(os.path.splitext(filePath)[0])+".png",maskImage)
        cv2.imwrite("Images/SortedImages("+str(index)+")/BoundedImages/"+os.path.basename(os.path.splitext(filePath)[0])+".png",image1)
            
        #interferenceMask = testDirectory((fileFolder+"/"+os.path.split(os.path.splitext(i)[0])[1]), interferenceMask, contours, eardrumArrays,  inputIndex, index)
        testArray(Objects, index, contourList, eardrumArrays, inputIndex)
        cv2.imwrite("Images/SortedImages("+str(index)+")/MaskedImages2/"+os.path.basename(os.path.splitext(filePath)[0])+".png",interferenceMask)
      
    takeAverageCords(eardrumArrays, index)




def testArray(Objects, index, contourList, eardrumArrays, inputIndex):
    global accuracyScore
    counter = 0
    for j in Objects:
            
            objectType = testNpArray(j, index, counter, inputIndex)
            if(objectType=="Interference"):
                pass
            elif(objectType=="Eardrum"):
                rect = cv2.boundingRect(contourList[counter])
                x,y,w,h = rect
                i=1
                accuracyString = accuracyScore.split(" ")[i].split("]")[0]
                while(accuracyString == ""):
                    i+=1
                    accuracyString = accuracyScore.split(" ")[i].split("]")[0]
                i=1
                accuracyNum = float(accuracyString)
                #print("Found 1! \n\n\n")
                if(accuracyNum<1e-15):
                    eardrumArrays[inputIndex].append([x,y,w,h])
            counter+=1
        
    
    


def testNpArray(image, index, num, inputIndex):
    global accuracyScore
    #currentImg =np.array(image)
    #img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.image.resize(image, (img_height, img_width))
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array,verbose = 0)
    score = tf.nn.softmax(predictions[0])
    #orderNum = ((int(os.path.basename(j).split('-')[1].split('.')[0])))
    #print("Testing np array")
    if CLASS_NAMES[np.argmax(score)] == 'Eardrum':
        tf.keras.utils.save_img("Images/SortedImages("+str(index)+")/Eardrum/"+str(inputIndex)+"-"+str(num)+".png", image)
        #cv2.imwrite("Images/SortedImages("+str(index)+")/Eardrum/"+str(num)+".png", image) 
    if CLASS_NAMES[np.argmax(score)] == 'Interference':
        tf.keras.utils.save_img("Images/SortedImages("+str(index)+")/Interference/"+str(inputIndex)+"-"+str(num)+".png", image)
       
        #cv2.imwrite("Images/SortedImages("+str(index)+")/Interference/"+str(num)+".png", image) 
        #print(str(orderNum-1) + " out of "+str(len(acceptedContours)))
        #cv2.drawContours(maskedImage,newContours[int(acceptedContours[orderNum-1])-1],0,255,-1)
    #print(CLASS_NAMES[np.argmax(score)])
    accuracyScore = str(score)
    return CLASS_NAMES[np.argmax(score)]\
    



def testArray2(Objects, cv2Object, index, inputIndex, inputContours):
    global EarDrumLayers
    global accuracyScore
    global earDrumBoundingBoxes
    global EardrumCenters
    global totalContours
    
    counter = 0
    #iterationCounter = 0
    for j in Objects:
            
            objectType = testNpArray2(j, index, counter, inputIndex)
            if(objectType=="Interference"):
                pass
            elif(objectType=="Eardrum"):
                i=1
                accuracyString = accuracyScore.split(" ")[i].split("]")[0]
                while(accuracyString == ""):
                    i+=1
                    accuracyString = accuracyScore.split(" ")[i].split("]")[0]
                i=1

                rect = cv2.boundingRect(inputContours[counter])
                x,y,w,h = rect
                #print("Type: "+str(type(inputContours[counter])))
                EardrumCenters[inputIndex].append([x+(w/2), y+(h/2)])
                totalContours[inputIndex].append([inputContours[counter]])
                #print(EardrumCenters[inputIndex])
                #print(EardrumCenters[inputIndex])
                cv2.drawContours(cv2Object,[inputContours[counter]],0,(255,255,255),-1)
                #iterationCounter +=1
            counter+=1

    return cv2Object

    
        
    
    


def testNpArray2(image, index, num, inputIndex):
    global accuracyScore
    #currentImg =np.array(image)
    #img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.image.resize(image, (img_height, img_width))
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array,verbose = 0)
    score = tf.nn.softmax(predictions[0])
    #orderNum = ((int(os.path.basename(j).split('-')[1].split('.')[0])))
    #print("Testing np array")
    if CLASS_NAMES[np.argmax(score)] == 'Eardrum':
        tf.keras.utils.save_img("Images/SortedImages("+str(index)+")2/Eardrum/"+str(inputIndex)+"-"+str(num+1)+".png", image)
        #cv2.imwrite("Images/SortedImages("+str(index)+")/Eardrum/"+str(num)+".png", image) 
    if CLASS_NAMES[np.argmax(score)] == 'Interference':
        tf.keras.utils.save_img("Images/SortedImages("+str(index)+")2/Interference/"+str(inputIndex)+"-"+str(num+1)+".png", image)
       
    accuracyScore = str(score)
    return CLASS_NAMES[np.argmax(score)]









            
def takeAverageCords(cordArray, index):
    #print(cordArray)
    global inputs
    os.mkdir("Images/SortedImages("+str(index)+")/IndividualBoxes")
    Volumes = []
    countTotal = 0
    avgIndex = [[0] * 4]*417 
    for i in range(0, len(cordArray)):
        #print(i)
        avgX = 0
        avgY = 0
        avgX2 = 0
        avgY2 = 0
        matched = False
        for j in range(0,len(cordArray[i])):
            if (cordArray[i][j][0]<avgX or avgX == 0):
                avgX = cordArray[i][j][0]
            if (cordArray[i][j][1]<avgY or avgY == 0):
                avgY = cordArray[i][j][1]
            
            if (cordArray[i][j][2]+cordArray[i][j][0]>avgX2 or avgX2 == 0):
                avgX2 = cordArray[i][j][2]+cordArray[i][j][0]
            
            if (cordArray[i][j][3]+cordArray[i][j][1]>avgY2 or avgY2 == 0):
                avgY2 = cordArray[i][j][3]+cordArray[i][j][1]
        
            avgIndex[j][0]=int(avgX)
            avgIndex[j][1]=int(avgY)
            avgIndex[j][2]=int(avgX2)
            avgIndex[j][3]=int(avgY2)
        #avgIndex[i][3]=int(avgY2)
        for j in range(0, len(Volumes)):
            if(avgX == 0 and avgY ==0 and avgX2 == 0 and avgY2 == 0):
                matched=True
            if(matched == False):
                if(abs((avgX2)-(Volumes[j][2]))<75 and abs((avgY2)-(Volumes[j][3]))<75 and abs(avgX-Volumes[j][0])<75 and abs(avgY-Volumes[j][1])<75):
                    matched= True
                    if((avgX<Volumes[j][0] and avgX!=0)or Volumes[j][0]==0):
                        Volumes[j][0]=avgX
                    if((avgY<Volumes[j][1] and avgY!=0)or avgY==0):
                        Volumes[j][1]=avgY
                    if(avgX2>Volumes[j][2]):
                        Volumes[j][2]=avgX2
                    if(avgY2>Volumes[j][3]):
                        Volumes[j][3]=avgX2
                    Volumes[j][4]+=1
        #print(matched)
        if(matched == False):
            Volumes.append([avgX,avgY,avgX2,avgY2,1])
    #print(Volumes)

    
    
    count = 1
    maxNum = 0
    area = 0
    maxXIndex = 0
    maxYIndex = 0
    MaxX2 = 0
    MaxY2 = 0
    #for j in range(0, len(Volumes)):
        #print(str(j)+" "+str(Volumes[j])+"\n\n")
    for j in range(0, len(Volumes)):
        if (Volumes[j][4]>3):
            if(Volumes[maxXIndex][0]>Volumes[j][0]and Volumes[j][0]!= 0 and Volumes[j][4]>3):
                maxXIndex = j
            if(Volumes[maxXIndex][0]==0):
               maxXIndex = j 

            if(Volumes[maxYIndex][1]>Volumes[j][1]and Volumes[j][1]!= 0 and Volumes[j][4]>3):
                maxYIndex = j
            if(Volumes[maxYIndex][1]==0):
               maxYIndex = j 

            if(MaxX2<Volumes[j][2] and Volumes[j][4]>3):
                MaxX2 = Volumes[j][2]
                
                
            if(MaxY2<Volumes[j][3] and Volumes[j][4]>3):
                

                MaxY2 = Volumes[j][3]
                #if (MaxY2>1500):print(j)
            count+=1
    avgX = Volumes[maxXIndex][0]
    avgY = Volumes[maxYIndex][1]
    avgX2 = MaxX2
    avgY2 = MaxY2
    if (avgX<0):avgX=0+20
    if (avgY<0):avgY=0+30
    if (avgX2>=417):avgX2=(417-20)
    if (avgY2>=2048):avgY2=(2048-30)

    
    count=1
    for j in list((inputs.glob("*.png"))):
        img = Image.open(str(j))
        name = os.path.split(str(j))[1]
        nonExtName = os.path.splitext(name)[0]
        
        croppedImg = img.crop((int(avgX-20),int(avgY-30),int(avgX2+20),int(avgY2+30)))
        #print(name)
        croppedImg.save("Images/SortedImages("+str(index)+")/CroppedImages/"+name)
        cvimg = cv2.imread(str(j))
        cvimg2 = cv2.imread(str(j))
        cvimg2 = cv2.rectangle(cvimg2, (int(avgIndex[count-1][0]),int(avgIndex[count-1][1])),(int(avgIndex[count-1][2]),int(avgIndex[count-1][3])),(0,255,0),0)
       
        newCvimg = cv2.rectangle(cvimg, (int(avgX-20),int(avgY-30)),((int(avgX2+20)),int(avgY2+30)),(0,255,0),2)
        cv2.imwrite("Images/SortedImages("+str(index)+")/IndividualBoxes/"+str(name), cvimg2)
        cv2.imwrite("Images/SortedImages("+str(index)+")/AvgImages/"+str(name), newCvimg)
        count+=1
        

        
    return [int(avgX), int(avgY), int(avgX2), int(avgY2)]






def secondTest(cropLocation, index):
    global EarDrumLayers
    contourX = 10
    contourY = 10
    thresholdInt = 60
    blurInt = 6
    inputs = Path(cropLocation)
    newOutputLocation = 'Images/EdgeDetectedCropped('+str(index)+')2/'
    
    eardrumArrays = [[] for _ in range(417)]
    for i in list(inputs.glob('*.png')):
        
        currentContours = []
        
        inputIndex = int(str(os.path.split(os.path.splitext(i)[0])[1]).replace('Layer',''))-1
        #(inputIndex)
        try:
            os.mkdir(newOutputLocation+str(os.path.split(os.path.splitext(i)[0])[1]))
        except:
            pass
        filePath = str(i)
        
        image1 = cv2.imread(filePath) 
        
        counter = 0
        image1Gray = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY )
        ret, image1Thresh = cv2.threshold(image1Gray, thresholdInt, 255, cv2.THRESH_BINARY)
        blurred = cv2.blur(image1Thresh, (blurInt,blurInt))

        contours, _ = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cropped_image = image1
        interferenceMask = image1.copy()
        #cv2.imwrite(inputIndex, maskImage)
        
        Objects = []
        counter = -1
        
        for c in contours:
            counter+=1
            rect = cv2.boundingRect(c)
            if rect[2] < contourX or rect[3] < contourY: continue
            x,y,w,h = rect
            image3 = image1.copy()
            cropped_image = image1[y:y+h, x:x+w]
            cv2.imwrite(newOutputLocation+str(os.path.split(os.path.splitext(i)[0])[1])+'/'+os.path.basename(os.path.splitext(filePath)[0])+"-"+str(counter)+".png",cropped_image)
            
            
            cropped_image = image1[y:y+h, x:x+w]
            
            image1Array = cropped_image
            img_tf = tf.convert_to_tensor(image1Array)
            Objects.append(img_tf)
            currentContours.append(c)



        cv2.imwrite("Images/SortedImages("+str(index)+")2/MaskedImages2/"+os.path.basename(os.path.splitext(filePath)[0])+".png",image1)
                

        interferenceMask = testArray2(Objects, image3, index, inputIndex, currentContours)
        cv2.imwrite(newOutputLocation+os.path.basename(os.path.splitext(filePath)[0])+".png",interferenceMask)
        

def averageEardrums(index, tiffNum):
    global EardrumCenters
    global totalContours
    lowerBound = index-5
    upperBound = index+5
    changeMade = False
    if(lowerBound<0):
        lowerBound = 0
    if(upperBound>416):
        upperBound = 416
    currentImg = cv2.imread("Images/EdgeDetectedCropped("+str(tiffNum)+")2/Layer"+str(index+1)+".png")
    
    #print(str(lowerBound+1)+", "+str(index+1)+", "+str(upperBound+1))
    #print(EardrumCenters[index])
    for i in range(lowerBound, upperBound+1):
        for m in range(0, len(EardrumCenters[i])):
            markFound = False
            for j in range(0, len(EardrumCenters[index])):
                if(markFound == False):
                    [x1,y1] = EardrumCenters[i][m]
                    [x2,y2] = EardrumCenters[index][j]
                    #print(str(math.hypot(x1, y1))+ ", "+str(math.hypot(x2, y2)))
                    if(abs(math.hypot(x1, y1)-math.hypot(x2, y2))<50):
                        #print("Problem with "+str(x1)+" and "+ str(y1))
                        markFound = True                                              
            if(markFound == False):
                #print("Issue at "+str(index+1)+", it's missing a TM from "+ str(i+1)) 
                changeMade = True
                #print(totalContours[index])
                cv2.drawContours(currentImg,totalContours[i][m],0,(255,255,255),-1)
    if(changeMade):
        #print("Rewriting Image "+ str(index+1))
        cv2.imwrite("Images/EdgeDetectedCropped("+str(tiffNum)+")2/Layer"+str(index+1)+".png", currentImg)


def EardrumFinder(tiffNum):
    EarDrumLayers.sort()
    inputs = "Images/SortedImages("+str(tiffNum)+")/CroppedImages/"
    for i in list(Path(inputs).glob('*.png')):
        currentVal = 0
        #print(i)
        count = int(str(os.path.split(os.path.splitext(i)[0])[1]).replace('Layer',''))
            #print(coordArray[0])
        Image = cv2.imread(str(i))
        averageEardrums(count-1, tiffNum)
        interferenceMask = cv2.imread("Images/EdgeDetectedCropped("+str(tiffNum)+")2/Layer"+str(count)+".png")
        interferenceMask = cv2.cvtColor(interferenceMask, cv2.COLOR_BGR2GRAY )
        _, interferenceMask = cv2.threshold(interferenceMask, 200, 255, cv2.THRESH_BINARY)
        currentVal+=1
        res = cv2.bitwise_and(Image,Image,mask = interferenceMask)
        cv2.imwrite("Images/BoxTests2("+str(tiffNum)+")/"+str(count)+".png", res)

            











CLASS_NAMES = ['Eardrum', 'Interference']
inputFolder = "Inputs"
tiffNum = 1
for i in os.listdir(inputFolder):
    EarDrumLayers = []
    earDrumBoundingBoxes = []
    #print(str(os.getcwd())+'\Models\\1st Modelset')
    model = keras.models.load_model(str(os.getcwd())+'\Models\\1st Modelset')
        

    data_dir = Path("Inputs\\"+i)
    if(os.path.isdir(data_dir)==False):
        continue


    inputs = data_dir
    print("Starting Cropping and Recognizing folder "+str(i))

    fileFolder = "Images/EdgeDetectedCropped("+str(i)+")/"

    dataset_url = "file:"+str(os.path.abspath('./'))[2:]


 




    makeDelDirs(tiffNum)

    print("Cropping All Images for folder "+str(i))

    runMain(tiffNum)

    #print("Compressing Images #"+str(i))
    #ThresholdContour.thresholdCutCompress(i)

  
    print("Starting Second Round on folder "+str(i))
    #model = keras.models.load_model('Models/2nd Modelset')
    model = keras.models.load_model('Models/Model2.2')
    secondTest("Images/SortedImages("+str(tiffNum)+")/CroppedImages", tiffNum)

    #timesRun = 0
    EardrumFinder( tiffNum)
    print("Starting Tiffing #"+str(i))
    PngtToTiff.pngToTiff("Images/BoxTests2("+str(tiffNum)+")/", tiffNum, i)
    tiffNum+=1
    shutil.rmtree("Inputs/"+str(i))
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print("Done")