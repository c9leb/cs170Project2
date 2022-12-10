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
distance_cache = {}


def dist(obj, data, fts):
    distance = 0
    match = (obj, data, frozenset(fts))
    for key, value in distance_cache.items():
        if key == match:
            return distance_cache[key]
    for i in fts:
        distance += (obj[i-1] - data[i-1])**2
    return np.sqrt(distance)
        
        
def feature_search_demo(data):
    
    current = []
    best_set = []
    best_best_acc = 0
    print("\nBeginning search...\n")
    for i in range(1, data.shape[1]):
        print(f"On level {i} of the search tree")
        addedft = []
        bestacc = 0
        for j in range(1, data.shape[1]):
            if(j not in current):
                print(f"  Consider adding feature {j}")
                
                acc = cross_validation(data, current, j)
                if acc > bestacc:
                   bestacc = acc
                   addedft = j
        
        current.append(addedft)          
        print(f"\nOn level {i} I added the feature {addedft} to the feature set which is now {current}")    
        print(f"The feature had an accuracy of {round(bestacc*100, 3)}%\n")
        if bestacc > best_best_acc:
            best_best_acc = bestacc
            best_set.append(addedft)
        
    print(f"\nBest feature set is {best_set} with an accuracy of {round(best_best_acc*100, 3)}%")
                       
def backwards_search_demo(data):
    
    current = []
    best_set = []
    best_best_acc = 0
    for i in range(1, data.shape[1]):
        current.append(i)
        best_set.append(i)
        
    print("\nBeginning search...\n")
    for i in range(1, data.shape[1]):
        print(f"On level {i} of the search tree")
        addedft = 0
        bestacc = 0
        for j in range(1, data.shape[1]):
            if(j in current):
                print(f"  Consider removing feature {j}")
                
                acc = bw_cross_validation(data, current, j)
                if acc > bestacc:
                   bestacc = acc
                   addedft = j
        
        current.remove(addedft)          
        print(f"\nOn level {i} I removed the feature {addedft} to the feature set which is now {current}")    
        print(f"The feature had an accuracy of {round(bestacc*100, 3)}%\n")
        if bestacc > best_best_acc:
            best_best_acc = bestacc
            best_set.remove(addedft)
        
    print(f"\nBest feature set is {best_set} with an accuracy of {round(best_best_acc*100, 3)}%")
                       


def cross_validation(data, current, addedft):
    classified = 0
    curr = 0
    curr = current[:]
    curr.append(addedft)
    for i in range(0, data.shape[0]):
        obj_to_classify = data[i, 1:]
        label_obj_to_classify = data[i, 0]
        
        nn_dist = 999999
        nn_loc = 999999
        for j in range(data.shape[0]):
            
            if j != i:
                distance = dist(obj_to_classify, data[j][1:], curr)
                if distance < nn_dist:
                    
                    nn_dist = distance
                    nn_loc = j
                    nn_label = data[nn_loc, 0]
                    
        if label_obj_to_classify == nn_label:
            classified+=1
    print(f"  Accuracy of set {curr}: {round(classified / data.shape[0], 2)}")        
    return classified / len(data)
                
def bw_cross_validation(data, current, addedft):
    classified = 0
    curr = 0
    curr = current[:]
    curr.remove(addedft)
    for i in range(0, data.shape[0]):
        obj_to_classify = data[i, 1:]
        label_obj_to_classify = data[i, 0]
        
        nn_dist = 999999
        nn_loc = 999999
        for j in range(data.shape[0]):
            
            if j != i:
                distance = dist(obj_to_classify, data[j][1:], curr)
                if distance < nn_dist:
                    
                    nn_dist = distance
                    nn_loc = j
                    nn_label = data[nn_loc, 0]
                    
        if label_obj_to_classify == nn_label:
            classified+=1
    print(f"  Accuracy of set {curr}: {round(classified / data.shape[0], 2)}")        
    return classified / len(data)
                
        
        
def main():
    datafile = input("Enter your data's file name to test: ")
    data = np.loadtxt(datafile)
    mode = int(input("Type the number of the algorithm you would like to run\n\t1. Forward Selection\n\t2. Backward Elimination\n>> "))
    print(f"This dataset has {data.shape[1]-1} features and {data.shape[0]} instances")
    if(mode == 1):
        feature_search_demo(data)
    if(mode == 2):
        backwards_search_demo(data)
    return

if __name__ == "__main__":
    main()