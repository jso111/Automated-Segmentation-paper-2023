In order to install the program you should follow these steps:

----Windows----

1. Ensure that you have python and virtual environment installed. 

2. Run the "Install.bat" file


-------------------------------------------------------------------------------------------------------------


In order to run the program you should follow these steps:

1. Drop all tiff files that you'd like to into the inputs folder. 1 example file has already been added to it.

2. Run the "Activate.bat" file

3. When the command prompt anounces that the program is done, you should find tiff file with the name of the original file with a .tiff extension inside of the Outputs folder

Remember that this code has been trained with small amounts of data as a trial, and that by training it with more data it will become more and more accurate.


-------------------------------------------------------------------------------------------------------------

In order to train more data, you will need to create a folder called "Training" inside this folder there must be two folders, Eardrum and Interference. You can go through
images that haven't been identified properly and put them in their respective folders within training. Then just zip Training as "Training.zip" This zip file should contain
the two aforementioned folders. After that you can follow these steps.

1. Run the "Train Data.bat" file

2. Your model will be saved in the Models folder and should be renamed but left in the Models folder so that it can be easily accessed and used within the python
script.

The classification of images is saved within the Images folder. SortedImages(N) and SortedImages(N)2 (N represents the number assigned to that tiff when running the script) 
have folders with Eardrum and Interference labels. These labels will not be 100% accurate and will require a small amount of manual sorting before they can be used to 
further train the model.