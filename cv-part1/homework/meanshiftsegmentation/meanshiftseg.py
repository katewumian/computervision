import numpy as np
import cv2
import random
import sys

'''Read an image.'''

#Input the path of the image and the name of the output image as two parameters.
if len(sys.argv) < 2:
    print('Error')
elif len(sys.argv) == 2:
    print('Please input both correct absolute path and the name of output image file.')
else:
    Input_Image = sys.argv[1]

#Input_Image = "C:/Users/sheaw/Desktop/cv/meanshiftsegmentation/input/tiger.png"
K = cv2.imread(Input_Image, 1)# The original image
#print(type(K))

print('load image in "original_image"')

row = K.shape[0]#row# of pixels
col = K.shape[1]#column# of pixels

print("row: " + str(row))
print("col: " + str(col))

J = row * col#the number of pixels of the image
Size = row, col, 3#size of the image, a tuple
R = np.zeros(Size, dtype=np.uint8)# The final image, [row][col][3(R/G/B feature)]
D = np.zeros((J, 5))#Feature space D，each row(pixel/point) corresponds to five dimensional features(R, G, B, x, y), [rows*cols][5]
arr = np.array((1, 3))#the array for saving three features for each pixel when converting image K into feature space D.

cv2.imshow("image", K)

counter = 0#pixels count

'''The motion is below a threshold or a finite number of steps has been taken. 
Here we use "iter" to define the threshold of whether to iterate to shift again.'''

'''iter: precision for measuring two close points as converging. Converging means the current mean 
   can represent the local neighbourhood.'''
iter = 1.0

threshold = 150#the radius of neighborhood is defined as an euclidean distance
current_mean_random = True#to save the current state of the mean to decide if it is needed to start a new point for searching
current_mean_arr = np.zeros((1, 5))#the five dimensional features of the current mean
below_threshold_arr = []#A set of all pixels whose euclidean distances to the mean are less than threshold


'''1. Convert the image K[row][col] into RGB together with pixel location Feature Space D. 
      The dimension of D is [rows*cols][5]. Each row of D is like (R, G, B, x, y). '''

for i in range(0, row):
    for j in range(0, col):
        arr = K[i][j]

        for k in range(0, 5):
            if (k >= 0) & (k <= 2):
                D[counter][k] = arr[k]
            else:
                if k == 3:
                    D[counter][k] = i
                else:
                    D[counter][k] = j
        counter += 1

'''6. (1)Instead of clustering all final values (modes) that are within a threshold, i.e., find the connected
      components in the end, we can color the neighbourhood at once we find the neighbourhoods that are already 
      satisfy some standards, i.e.,when in below_threshold_arr(neighbors) & mean_e_distance < iter.
      (2)By eliminating already labeled pixels, we can reduce iterating times.
      (3)Set the lower limit of iteration, that is, to stop the iteration when there are N points left, 
      but this method is commented out because it greatly affects accuracy.'''
while len(D) > 0:
    # print J
    print(len(D))
    '''5. Use a random subset of the pixels as starting points and find which component
          each unlabeled pixel belongs to, either by finding its nearest neighbor or by 
          iterating the mean shift until it finds a neighboring track of mean-shift values.
          
          Data structures:
          '''
    #If the current mean is found, select a random new row from the feature space and assigning it as the current mean arr.
    #Or, continue iterating skipping the random step.
    if current_mean_random:
        print("random++")
        current_mean = random.randint(0, len(D) - 1)
        for i in range(0, 5):
            current_mean_arr[0][i] = D[current_mean][i]

    '''2. Finding all the neighbours around the current mean array,
       by checking if the distance calculated is within the threshold. If yes taking those rows and adding
       them to a list called below_threshold_arr'''
    below_threshold_arr = []
    for i in range(0, len(D)):
        # print "Entered here"
        ecl_dist = 0
        color_total_current = 0
        color_total_new = 0
        # Calculating the eucledian distance of the current mean row(To the first time was a
        # randomly selected row) with all the other rows
        for j in range(0, 5):
            ecl_dist += ((current_mean_arr[0][j] - D[i][j]) ** 2)

        ecl_dist = ecl_dist ** 0.5

        if (ecl_dist < threshold):
            below_threshold_arr.append(i)
            # print "came here"

    '''2. Compute the mean of all five features as an array of the current_mean's neighbours.'''
    mean_R = 0
    mean_G = 0
    mean_B = 0
    mean_i = 0
    mean_j = 0
    current_mean = 0
    mean_col = 0

    # For all the rows found and placed in below_threshold_arr list, calculating the average of
    # Red, Green, Blue and index positions.

    for i in range(0, len(below_threshold_arr)):
        mean_R += D[below_threshold_arr[i]][0]
        mean_G += D[below_threshold_arr[i]][1]
        mean_B += D[below_threshold_arr[i]][2]
        mean_i += D[below_threshold_arr[i]][3]
        mean_j += D[below_threshold_arr[i]][4]

    mean_R = mean_R / len(below_threshold_arr)
    mean_G = mean_G / len(below_threshold_arr)
    mean_B = mean_B / len(below_threshold_arr)
    mean_i = mean_i / len(below_threshold_arr)
    mean_j = mean_j / len(below_threshold_arr)

    '''Finding the distance of these average values with the current mean and comparing it with iter'''

    mean_e_distance = ((mean_R - current_mean_arr[0][0]) ** 2 + (mean_G - current_mean_arr[0][1]) ** 2 + (
                mean_B - current_mean_arr[0][2]) ** 2 + (mean_i - current_mean_arr[0][3]) ** 2 + (
                                   mean_j - current_mean_arr[0][4]) ** 2)

    mean_e_distance = mean_e_distance ** 0.5

    nearest_i = 0
    min_e_dist = 0
    counter_threshold = 0
    # If less than iter, find the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
    # This is because mean_i and mean_j could be decimal values which do not correspond
    # to actual pixel in the Image array.

    if mean_e_distance < iter:
        # print "Entered here"

        new_arr = np.zeros((1, 3))#Three features R G B of the mean pixel
        new_arr[0][0] = mean_R
        new_arr[0][1] = mean_G
        new_arr[0][2] = mean_B

        '''4. Color(Label) all the points that are connected, whose distances with the current 
           mean are within a threshold(in below_threshold_arr) and mean_e_distance < iter, 
           with the mean color of the neighbourhood.'''

        for i in range(0, len(below_threshold_arr)):
            R[int(D[below_threshold_arr[i]][3])][int(D[below_threshold_arr[i]][4])] = new_arr

            '''Also now don't use those rows that have been colored once.'''
            D[below_threshold_arr[i]][0] = -1

        current_mean_random = True
        new_D = np.zeros((len(D), 5))
        #print("len: " + str(len(D)))
        counter_i = 0

        #Apply all that don't have -1 value to a new_D array.
        for i in range(0, len(D)):
            if D[i][0] != -1:
                new_D[counter_i][0] = D[i][0]
                new_D[counter_i][1] = D[i][1]
                new_D[counter_i][2] = D[i][2]
                new_D[counter_i][3] = D[i][3]
                new_D[counter_i][4] = D[i][4]
                counter_i += 1

        D = np.zeros((counter_i, 5))#form a new empty array D with row number of counter_i
        print("New len: " + str(len(D)))

        for i in range(0, counter_i):
            D[i][0] = new_D[i][0]
            D[i][1] = new_D[i][1]
            D[i][2] = new_D[i][2]
            D[i][3] = new_D[i][3]
            D[i][4] = new_D[i][4]

    else:
        '''3. Make the shift:       
              Replace the current value with the mean array and iterate if the eucledian distance between 
              the current mean pixel and the new mean pixel is above or equal to a threshold "iter".'''
        current_mean_random = False#Telling next iteration not to generate a new random starting point.

        current_mean_arr[0][0] = mean_R
        current_mean_arr[0][1] = mean_G
        current_mean_arr[0][2] = mean_B
        current_mean_arr[0][3] = mean_i
        current_mean_arr[0][4] = mean_j

        # cv2.imwrite("image"+ str(len(below_threshold_arr)) +".png", R)

    # if(len(D) >= 40000):
    #     break
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('finalImage', R)
cv2.imwrite('output/finalImage' + str(sys.argv[2]) + str(threshold) + '.png', R)
cv2.waitKey(0)
cv2.destroyAllWindows()
