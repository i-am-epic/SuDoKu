
from ast import main
import cv2
import numpy as np
from slicer import slicer

from boxidentify import BoxIdentify
# start with min thresh for black and white image
thresh = 85 #70-150
# define a threshold, 128 is the middle of black and white in grey scale

if __name__ == "__main__":
        
    print("Finding Sudoku BOX ..................")
    boxIdfy = BoxIdentify(thresh)
    boxIdfy.SudokuIdentifier()
    print("Slicing the Sudoku BOX ..................")
    slicer()
