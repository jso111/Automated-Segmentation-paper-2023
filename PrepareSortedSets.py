import os
import shutil
import cv2
import random
import imutils

interferenceNum = 0
eardrumNum = 0
factorN = 2
folderName = "Training Data"
# Function to rename multiple files
try:
    shutil.rmtree(folderName+"/Training")
except:
    pass
os.mkdir(folderName+"/Training")
def main(num, folder):
    global interferenceNum
    global eardrumNum
    try:
        i = 0
        path= folderName+"/Set"+str(num)+"/"+ folder + "/"
        newPath = folderName+"/Training/"+folder+"/"
        try:
            os.mkdir(newPath)
        except:
            pass
        #print(path)
        for filename in os.listdir(path):
            #print("hi")
            if (folder=="Eardrum"):
                eardrumNum+=1
            else:
                interferenceNum+=1
            my_dest =str(num)+ "-" + folder+"-New-" + str(i) + ".jpg"
            my_source =path + filename
            #new_source = newPath+filename
            my_dest =newPath + my_dest
            # rename() function will
            # rename all the files
            shutil.copy(my_source,my_dest)
            i += 1
    except Exception as e:
        print("Issue on "+ str(num))
        print(e)
        


def pickOnlyN(path,N):
    splitPath = (os.path.normpath(path)).split(os.sep)
    i = 1
    
    try:
        os.mkdir(folderName+"/Training/NOutputs-" + splitPath[len(splitPath)-1])
    except:
        print("Fail")
        pass
    for fileName in os.listdir(path):
        chance = random.randint(0, N-1)
        inputImage = cv2.imread(path+fileName)
        if (chance == 0):
            i+=1
            cv2.imwrite(folderName+"/Training/NOutputs-" + splitPath[len(splitPath)-1] +"/"+str(i)+"_"+fileName, inputImage) 
        
    #print(i)




try:
    os.mkdir(folderName+"/Training/RotatedOutputs")
except:
    pass

def rotateImage(path,numRotations):
    splitPath = (os.path.normpath(path)).split(os.sep)
       
    i = 1
    try:
        shutil.rmtree(folderName+"/Training/RotatedOutputs/" + splitPath[len(splitPath)-1])
    except:
        pass
    os.mkdir(folderName+"/Training/RotatedOutputs/" + splitPath[len(splitPath)-1])
    for fileName in os.listdir(path):
        
       
        for j in range(1, numRotations):
            
            inputImage = cv2.imread(path+fileName)
            angle = random.randint(0, 360)
            rotated = imutils.rotate_bound(inputImage, angle)
            cv2.imwrite(folderName+"/Training/RotatedOutputs/" + splitPath[len(splitPath)-1] + "/"+str(j)+"_"+fileName, rotated) 
                    
       

        cv2.imwrite(folderName+"/Training/RotatedOutputs/" + splitPath[len(splitPath)-1] + "/"+str(numRotations)+"_"+fileName, inputImage) 
        
        i+=1

# Make sure to add compression renaming and deleting before next run

print("Starting")
for j in range(2,8):
    main(j, "Eardrum")    
    main(j, "Interference")
    #main(j, "Maybe")
print("Finished renaming")
print("Finished picking N factor")
print(str(eardrumNum)+", "+str(interferenceNum))
#pickOnlyN(folderName+"/Training/Interference/",int(interferenceNum/eardrumNum))
overfitFactor = int(interferenceNum/(eardrumNum))
print(overfitFactor)
rotateImage(folderName+"/Training/Eardrum/", 2)
print("Finished rotating images")
print("Done")