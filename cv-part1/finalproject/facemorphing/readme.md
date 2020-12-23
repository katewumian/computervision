# FaceMorphing

## About

## morph one face to another using matplotlib.pyplot.Delaunay, dlib.shape_predictor, and shape_predictor_68_face_landmarks.dat model to detect features automatically or manually and morph faces.

## Direction:

_Open_ the project in PyCharm, _install_ the libraries in **requirements.txt**, adjust the parameters, _run_ the **main.py** file,

The following parameters are mainly modified before running the code:

1) _FRAME_COUNT_ = 45 # This parameter represents the number of frames of the image that will eventually be used to generate the animation, 45 was selected for the experiment.

2) _FRAME_PER_SECOND_ = 30, 30 frames per Second, parameter for composing a GIF.

3) _FRAME_TO_PATH_ = "wu_to_su_", compose the common part of names of Frames, also for the name of the file for UcKEY Point locations.

4) _GIF_SOURCE_PATH_ = _FRAME_TO_PATH_

5) _IMG1_ = "./input/wuyanzu.jpg"

6) _IMG2_ = "./input/ sudaqiang.jpg"

7) _IF_RECAL_ = False #False means importing stored_points for _IMG1_ and _IMG2_ from a known file;True represents recalculation of _IMG1_ and _IMG2_ feature points.

8) _IF_AUTO_ = True #When _IF_RECAL_ == True, _IF_AUTO_ == True means automatic detection of feature points, and when _IF_AUTO_ == False means manual selection of feature points.

9) _GIF_NAME_ = "./output/wu_to_su.gif"

10) _POINT_COUNT_ = 70 #Upper limitation for points number when manually making correspondences.

11) _PREDICTOR_ = "**shape_predictor_68_face_landmarks.dat**" is the model of face detection in Dlib, which is used for auto-detect situation. Can download the model from the link in the file "**model_download_link.txt**" attached in the **input** directory and put the file into the "**input**" directory. (Too large for uploading.) 

Other notes: In order to pick a point on the image, the image should not appear in the Tool Window (PyCharm) if the manual-way is selected, that is, when the parameters "_IF_RECAL_ = True" and "_IF_AUTO_ =False" are set. Take PyCharm (Windows) as an example. In **File**- **Setting**- **Tools**- **Python Scientific**, make sure you don't choose the "_Show plots in tool window_" option.
