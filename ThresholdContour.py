import cv2
import os
from pathlib import Path
import shutil

def thresholdCutCompress(index):

    inputs = Path("Images/SortedImages("+str(index)+")/CroppedImages/")
    outputs = Path("Images/SortedImages("+str(index)+")/CroppedImages/OutputMasks")
    thresholdInt = 30
    blurInt = 2
    contourX = 30
    contourY = 30

    try:
        shutil.rmtree(outputs)
    except:
        pass
    os.mkdir(outputs)
    os.mkdir(str(outputs)+"/BoundedImages/")
    os.mkdir(str(outputs)+"/EdgeDetectedCropped/")

    for i in list(inputs.glob('*.png')):
        #print(i)
        inputIndex = int(str(os.path.split(os.path.splitext(i)[0])[1]).replace('Layer',''))
        #print(inputIndex)
        filePath = str(i)

        image1 = cv2.imread(filePath)
        
        counter = 0

        image1Gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY )

        ret, image1Thresh = cv2.threshold(image1Gray, thresholdInt, 255, cv2.THRESH_BINARY)
        blurred = cv2.blur(image1Thresh, (blurInt,blurInt))

        contours, _ = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cropped_image = image1
        maskImage = image1.copy()
        image2 = image1.copy()
        interferenceMask = image1.copy()
        #print(len(contours[1]))
        goodContours = []
        counter = -1
        for c in contours:
            counter+=1
            rect = cv2.boundingRect(c)
            if rect[2] < contourX or rect[3] < contourY:
                #cv2.drawContours(image1,[c],0,0,0)
                pass
            else:
                x,y,w,h = rect
                cv2.drawContours(maskImage,[c],0,255,-1)
                cropped_image = image2[y:y+h, x:x+w]
                cv2.rectangle(image1,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imwrite(str(outputs)+'/EdgeDetectedCropped/'+os.path.basename(os.path.splitext(filePath)[0])+"-"+str(counter)+".png",cropped_image)
    
            
        cv2.imwrite(str(outputs)+"/"+str(os.path.basename(os.path.splitext(filePath)[0]))+".png",maskImage)
        cv2.imwrite(str(outputs)+"/BoundedImages/"+os.path.basename(os.path.splitext(filePath)[0])+".png",image1)
            
        #cv2.imwrite('Images/SortedImages/BoundedImages/'+os.path.basename(os.path.splitext(filePath)[0])+".png",image1)
            
    # cv2.imwrite('Images/SortedImages/MaskedImages2/'+os.path.basename(os.path.splitext(filePath)[0])+".png",interferenceMask)
        #break
    try:
        shutil.rmtree("Training stuff/Normal/CroppedImages"+str(index)+".zip")
    except:
        pass
   
    shutil.make_archive("Training stuff/Normal/CroppedImages"+str(index), 'zip', inputs)