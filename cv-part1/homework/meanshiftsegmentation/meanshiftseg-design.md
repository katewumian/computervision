#  Image-Segmentation-Mean-Shift Design Document

**Name**: Wu, Mian

**Student ID**: 201753004

**Sequence Number**: 32

## Homework Requirement

1. Convert your image to L*a*b space, or keep the original RGB colors, and augment
them with the pixel (x; y) locations.

2. For every pixel (L; a; b; x; y), compute the weighted mean of its neighbors using either
a unit ball (Epanechnikov kernel) or finite-radius Gaussian, or some other kernel of
your choosing. Weight the color and spatial scales differently, e.g., using values of
(hs; hr;M) = (16; 19; 40) as shown in Figure 5.18.

3. Replace the current value with this weighted mean and iterate until either the motion is
below a threshold or a finite number of steps has been taken.

4. Cluster all final values (modes) that are within a threshold, i.e., find the connected
components. Since each pixel is associated with a final mean-shift (mode) value, this
results in an image segmentation, i.e., each pixel is labeled with its final component.

5. (Optional) Use a random subset of the pixels as starting points and find which component
each unlabeled pixel belongs to, either by finding its nearest neighbor or by
iterating the mean shift until it finds a neighboring track of mean-shift values. Describe
the data structures you use to make this efficient.

6. (Optional) Mean shift divides the kernel density function estimate by the local weighting
to obtain a step size that is guaranteed to converge but may be slow. Use an alternative
step size estimation algorithm from the optimization literature to see if you can
make the algorithm converge faster.

## Data Structures

#### Fields

int _row_: row# of pixels

int _col_: column# of pixels

int _J_: the number of pixels of the image

tuple _Size_: size of the image

numpy.ndarray _K_: the original image, represented as an array of [_row_][_col_][3] dimensions

numpy.ndarray _R_: the final image, represented also as an array of [_row_][_col_][3] dimensions

numpy.ndarray _D_: feature space D converted from the original image，each pixel (R, G, B) augmented with locations (x, y), forming a new 2-D array with _J_ rows and five dimensional features(R, G, B, x, y), size [rows*cols][5]

numpy.ndarray _arr_: a 1*3 temporary array for saving three features for each pixel when converting image K into feature space D

numpy.ndarray _current_mean_arr_: an array for saving five dimensional features of the current mean

float _iter_: precision for measuring two close points as converging

int _threshold_: the radius of neighborhood, defined as an euclidean distance

boolean _current_mean_random_: to save the current state of the mean to decide if it is needed to start a new point for searching

list _below_threshold_arr_: a set of all pixels whose euclidean distances to the mean are less than threshold

## Algorithms 

#### Neighbours

Use a unit ball(radius == threshold). The radius of the unit ball is defined as euclidean distance:

    distance^2 = ((mean_R - current_mean_arr[0][0]) ** 2 + (mean_G - current_mean_arr[0][1]) ** 2 + (mean_B - current_mean_arr[0][2]) ** 2 + (mean_i - current_mean_arr[0][3]) ** 2 + (mean_j - current_mean_arr[0][4]) ** 2)

#### Mean Shift
    
    random_true = True
    while True:
        if random_true:
            Generate a new random starting point.
        if mean_distance < iter:
            Label the neighbourhood.
        else:
            random_true = False
            Shift the mean.
        if end:
            break;

#### How to Make the Algorithm Converge Faster?

(1) We can reduce the number of shifts by defining the precision. That is, as long as the distance between two points reaches a certain degree of convergence, this round can be ended, i.e.,mean_e_distance < iter.

(2) After coloring the neighbourhood, we can exclude labeled pixels out and begin next iteration only within pixels left in the new feature space D.
