@echo off
call "Env/Scripts/activate.bat"
java SplitLayer
python PrepareSortedSets.py 

pause