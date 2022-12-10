import queue
from node import Node
import time
import numpy as np
import math

#       sleeping cat
#         |\      _,,,---,,_
#   ZZZzz /,`.-'`'    -.  ;-;;,_
#        |,4-  ) )-,_. ,\ (  `'-'
#       '---''(_/--'  `-'\_)  
#Used and modified Professor Eamonn Keogh's sample code

smalldata = np.loadtxt("./CS170_Small_Data__93.txt")
largedata = np.loadtxt("./CS170_Large_Data__15.txt")

def feature_search_demo(data):
    
    current = []
    for i in range(1, data.shape[1]-1):
        print(f"On level {i} of the search tree")
        addedft = []
        bestacc = 0
        
        for j in range(1, data.shape[1]-1):
            if(j in current):
                print(f"--Consider adding feature {j}")
                acc = cross_validation(data, current, j+1)
            
                if acc > bestacc:
                   bestacc = acc
                   addedft = j+1
        current[i] = addedft           
        print(f"On level {i} I added the feature {addedft} to current set")    
    return

def cross_validation(data, current, ft):
    classified = 0
    for i in range(1, data.shape[1]-1):
        obj_to_classify = data[i, 1:]
        label_obj_to_classify = data[i, 1]
        
        nn_dist = 999
        nn_loc = 999
        for j in range(1, data.shape[1]-1):
            
            if j != i:
                
                dist = math.sqrt(sum((obj_to_classify - data(j, 2:)), ))
                if dist < nn_dist:
                    
                    nn_dist = dist
                    nn_loc = j
                    nn_label = data(nn_loc, 1)
                    
        if label_obj_to_classify == nn_label:
            classified = classified + 1
    
    return classified / len(data)
                
        
        
def main():
    
    feature_search_demo(smalldata)
    
    return

if __name__ == "__main__":
    main()