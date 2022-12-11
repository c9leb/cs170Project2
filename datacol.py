import time
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

#       sleeping cat
#         |\      _,,,---,,_
#   ZZZzz /,`.-'`'    -.  ;-;;,_
#        |,4-  ) )-,_. ,\ (  `'-'
#       '---''(_/--'  `-'\_)  
#Used and modified Professor Eamonn Keogh's sample code

smalldata = np.loadtxt("./CS170_Small_Data__93.txt")
largedata = np.loadtxt("./CS170_Large_Data__15.txt")

#nn distance function using euclidean distance
def dist(obj, data, fts):
    distance = 0
    for i in fts:
        distance += (obj[i-1] - data[i-1])**2
    return np.sqrt(distance)
        
#feature search function
def feature_search_demo(data, df):
            
    #initializes the list of current features and best set of features
    current = []
    best_set = []
    best_best_acc = 0
    
    print("\nBeginning search...\n")
    
    #loops over features to consider
    for i in range(0, data.shape[1]):
        print(f"On level {i} of the search tree")
        addedft = []
        bestacc = 0
        
        if(i == 0):
            best_best_acc = cross_validation(data, current, 0)
            df2 = {'Current Feature Set': str(current), 'Accuracy': str(round(best_best_acc*100, 3))}
            df = df.append(df2, ignore_index = True) 
        else:
            #loops over possible features to add
            for j in range(1, data.shape[1]):
                if(j not in current):
                   #print(f"  Consider adding feature {j}")

                    #computing accuracy of the feature set with the added feature and compare to the current best accuracy
                   acc = cross_validation(data, current, j)
                   if acc > bestacc:
                       bestacc = acc
                       addedft = j
        
            current.append(addedft)
            df2 = {'Current Feature Set': str(current), 'Accuracy': str(round(bestacc*100, 3))}
            df = df.append(df2, ignore_index = True)          
            print(f"\nOn level {i} I added the feature {addedft} to the feature set which is now {current}")    
            print(f"The feature had an accuracy of {round(bestacc*100, 3)}%\n")
        
            #finds the best accuracy out of all feature sets computed
            if bestacc > best_best_acc:
                best_best_acc = bestacc
                best_set.append(addedft)
        
    print(f"done")
    return df
   
#backwards feature search function, very similar to the forward search function             
def backwards_search_demo(data, df):
    
    current = []
    best_set = []
    best_best_acc = 0
    
    #adds all features to the current set
    for i in range(1, data.shape[1]):
        current.append(i)
        best_set.append(i)
        
    print("\nBeginning search...\n")
    for i in range(0, data.shape[1]):
        #print(f"On level {i} of the search tree")
        addedft = 0
        bestacc = 0
        if(i == 0):
            best_best_acc = bw_cross_validation(data, current, 0)
            df2 = {'Current Feature Set': str(current), 'Accuracy': str(round(best_best_acc*100, 3))}
            df = df.append(df2, ignore_index = True) 
            #print('\n')
        else:
            for j in range(1, data.shape[1]):
                if(j in current):
                    #print(f"  Consider removing feature {j}")

                    acc = bw_cross_validation(data, current, j)
                    if acc > bestacc:
                       bestacc = acc
                       addedft = j
        
            #removes ft from current set
            current.remove(addedft)     
            df2 = {'Current Feature Set': str(current), 'Accuracy': str(round(bestacc*100, 3))}
            df = df.append(df2, ignore_index = True)          
            print(f"\nOn level {i} I removed the feature {addedft} to the feature set which is now {current}")    
            print(f"The feature had an accuracy of {round(bestacc*100, 3)}%\n")
            if bestacc > best_best_acc:
                best_best_acc = bestacc
                best_set.remove(addedft)
        
    print(f"done")
    return df
                       

#computes accuracy of the classifier
def cross_validation(data, current, addedft):
    classified = 0
    #copies current feature set and the added feature into the curr array
    curr = 0
    curr = current[:]
    if(addedft != 0):
        curr.append(addedft)
    
    #loop over all instances in the dataset
    for i in range(0, data.shape[0]):
        
        #object to classify, is row i with all features from row one on
        obj_to_classify = data[i, 1:]
        
        #label i
        label_obj_to_classify = data[i, 0]
        
        nn_dist = 999999
        nn_loc = 999999
        
        #loops through all instances and calculates nn distances
        for j in range(data.shape[0]):
            
            if j != i:
                distance = dist(obj_to_classify, data[j][1:], curr)
                if distance < nn_dist:
                    
                    nn_dist = distance
                    nn_loc = j
                    nn_label = data[nn_loc, 0]
        
        #calculates total classified labels            
        if label_obj_to_classify == nn_label:
            classified+=1
    #print(f"Time: {round(time2 - time1, 4)}\n")
    #print(f"  Accuracy of set {curr}: {round(classified / data.shape[0], 2)}")        
    return classified / len(data)

#essentially the same as forward except it will remove instead of appending features                
def bw_cross_validation(data, current, addedft):
    classified = 0
    curr = 0
    curr = current[:]
    if(addedft != 0):
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
    #print(f"  Accuracy of set {curr}: {round(classified / data.shape[0], 2)}")        
    return classified / len(data)
                
        
        
def main():
    df = pd.DataFrame(columns=['Current Feature Set', 'Accuracy'])
    df3 = pd.DataFrame(columns=['Current Feature Set', 'Accuracy'])
    data = largedata
    time1 = time.time()
    df = feature_search_demo(data, df)
    time2 = time.time()
    print(f"Time: {round(time2 - time1, 4)}\n")
    time1 = 0
    time2 = 0
    time1 = time.time()
    df3 = backwards_search_demo(data, df3)
    time2 = time.time()
    print(f"Time: {round(time2 - time1, 4)}\n")
    print(df)
    print(df3)
    df3.to_csv("largebw.csv")
    df.to_csv("large.csv")
    return

if __name__ == "__main__":
    main()