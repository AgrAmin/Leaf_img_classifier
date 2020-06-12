# A generic algorithm for preprocessing image datasets, in particular including tasks of contour extraction
# Preapring datasets for ml algorithms such as Knn. The code tested with couple of datasets (HWD, Leafs,...) and it
#works like charm with ML algorithms such as Knn.

#this code :
'''
1- dataset of images where each class in a folder (check the code for this maybe needs to chnage for each case)
2- each image should contain one object and a plain background

3- the images will be cropped and resised to a given size (default 80*80 , you can change it)
4- cropped images are grayscale images
5- The biggest cintour is extracted and fed directly to the clutering method (knn)

6- Futurs files in this rep will deal with more advanced methods to generate feature descriptors + feature selection

'''
import numpy as np
import cv2
import imutils
import os
from pathlib import Path

def x_cord_contour(contor):
    # outputs the x centroid coordinates

    if cv2.contourArea(contor) > 100:
    #print((cv2.contourArea(contor)))
        M = cv2.moments(contor)
    #print(M)
    return (int(M['m10'] / M['m00']))


def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed

    BLACK = [0, 0, 0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        # print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = (height - width) / 2
            pad=int(pad)
            # print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad,
                                                   pad, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = (width - height) / 2
            pad = int(pad)
            # print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0, \
                                                   cv2.BORDER_CONSTANT, value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    # print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square

def resize_to_pixel(dimensions,image):
    #resize image to specified dimension
    buffer_pix=4
    dimensions= dimensions - buffer_pix
    squared = image
    r = float(dimensions)/squared.shape[1]
    dim = (dimensions,int(squared.shape[0]*r))
    resized = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r, width_r=img_dim2.shape
    BLACK=[0,0,0]


def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions

    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = makeSquare(image)
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0, 0, 0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg

def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
	# return the image with the contour number drawn on it

	return image

def load_img(link):
    image=cv2.imread(link)
    cv2.imshow('imagee',image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayy',gray)
    cv2.waitKey(0)

    # Blur
    Blurred = cv2.GaussianBlur(gray, (5, 5),cv2.BORDER_DEFAULT)
    cv2.waitKey(0)
    # find edges
    edges = cv2.Canny(Blurred, 30, 150)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    #dilation
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=3)

    cv2.imshow('Dilation', dilation)
    cv2.waitKey(0)

    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Closing', closing)
    cv2.waitKey(0)
    # contours
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    orig = edges.copy()
    # loop over the (unsorted) contours and draw them
    orig = draw_contour(orig, cnts[-1], 1)
    (x, y, w, h) = cv2.boundingRect(cnts[0])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 7)
    print(w)
    print(h)

    cv2.imshow("Unsorted", image)
    cv2.waitKey(0)
    for (i, c) in enumerate(cnts):
        orig = draw_contour(orig, c, i)
        (x, y, w, h) = cv2.boundingRect(cnts[i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 7)
    # show the original, unsorted contour image
    cv2.imshow("Unsorted", image)
    cv2.waitKey(0)
    '''
    # Sort contours by their cordinates
    contour = sorted(contour, key=x_cord_contour, reverse=False)
    cv2.imshow('Conout', contour(1))
    cv2.waitKey(0)
    '''
    return edges

def load_cropimg (link):
    img_size=80
    image = cv2.imread(link)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    Blurred = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    # find edges
    edges = cv2.Canny(Blurred, 30, 150)
    # dilation
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=5)

    #cv2.imshow('Dilation', dilation)
    #cv2.waitKey(0)

    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    # contours
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    (x, y, w, h) = cv2.boundingRect(cnts[0])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 7)
    crop_img = gray[y:y + h, x:x + w]
    #cv2.imshow("cropped", crop_img)
    res=resize_to_pixel(img_size,crop_img)
    #cv2.imshow('res',res)
    #cv2.waitKey(0)
    #print(res.shape)
    return res#crop_img

#test load random image from dataset image
#load_cropimg('l5nr008.TIF')



#list direcotries / classes

#extract classes out folder names
#That's my directory in my local computer ...
arr = os.walk(r"C:\Users\Asus\Desktop\H\2 H\mars 18\T doc\datasets\Swedish dataset\Nouveau dossier")
list_dir=[x[0] for x in arr]
#listr.split(sep='\\')
class_names= [list_dir[class_name].split(sep='\\')[-1] for class_name in range (1,len(list_dir)-1)]
print(class_names)

#forming dataset
ratio = 0.8#ratio train / test data

print('done')
train_x= []# also possible np.empty((0,80*80)) but working with lists is faster
test_x = []#np.array([])

for cl in range (1,len(list_dir)-1):
    ar = os.listdir(list_dir[cl])
    print(list_dir[cl])
    cells = [load_cropimg(list_dir[cl] + "\\" + i).reshape(-1, 80 * 80).astype(np.float32) for i in ar]
    train = np.array(cells).squeeze()

    #print("more info about train_x"+str(np.shape(train_x)))
    train_x.append(train[:int(ratio * len(ar)),:])
    test_x.extend(train[int(ratio * len(ar)):, :])
    #print("shape of test x "+str(np.shape(test_x)))

    print(train.shape[0])
    #for i in range(0,int(ratio * len(ar))):
     #   labelsT.extend(str(cl))
    #for i in range(0,train.shape[0]-int(ratio * len(ar))):
     #   labelst.extend(str(cl))


#Labels associated with training /testing sets
labelsT=[]
labelst=[]
for cl in range(1, len(list_dir) - 1):
    ar = os.listdir(list_dir[cl])
    for i in range(0,int(ratio * len(ar))):
        labelsT.extend([[cl]])
    for i in range(0,len(ar)-(int(ratio * len(ar)))):
        labelst.extend([[cl]])



print('We are here in second part baby')
print(np.shape(train_x))
Train=(np.array(train_x)).reshape(-1,80*80)
print(Train.shape)
print(np.shape(test_x))
Test= (np.array(test_x)).reshape(-1,80*80)
print(Train.shape)
print(Test.shape)
print('Hmmm begin knn')

##
# the last part Training and evaluation

#knn train data

LabelsT= np.array(labelsT)
Labelst= np.array(labelst)
print(LabelsT.shape)

knn= cv2.ml.KNearest_create()#cv2.KNearest()
knn.train(Train,cv2.ml.ROW_SAMPLE,LabelsT)
ret,result,neighbours,distance= knn.findNearest(Test,k=3)

matches=result==Labelst
correct=np.count_nonzero(matches)
accuracy=correct*(100.0/result.size)
print("accuracy is = %.2f"%accuracy+"%")
print("neighbours: ", neighbours,"\n")
print("distance: ", distance)
