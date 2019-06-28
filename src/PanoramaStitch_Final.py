import cv2
import numpy as np
import random
from os import listdir
from os.path import isfile, join
import argparse
import sys
import matplotlib.pyplot as plt

# helper function, show images
def show_image(img):
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    cv2.imshow('image', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

# Find correspondences between images
def findMatchPair(dscrptor1, dscrptor2, kp1, kp2):
    result=list()
    superlist=list()
    dMatchList=list()
    distList=list()
    xDistList = list()
    yDistList = list()

    # load all coordinates
    pts1 = np.asarray([[p.pt[0],p.pt[1]] for p in kp1 ])
    pts2 = np.asarray([[p.pt[0],p.pt[1]] for p in kp2 ])

    # find the correspondence
    for i, vector1 in enumerate(dscrptor1):
        curr=list()
        for vector2 in dscrptor2:
            dist = np.sqrt(np.sum(np.square(vector1-vector2)))
            curr.append(dist)

        # for each points in descriptor1 , find the closest matching descriptor in 2
        j = np.argmin(curr)
        minDist = min(curr)

        # calculate the distance between the matching pointers
        coordDist = ((pts1[i][0] - pts2[j][0])**2+(pts1[i][1] - pts2[j][1])**2)**0.5

        # record each pair of the mathing points and the corresponding distance
        xDistList.append(pts1[i][0] - pts2[j][0])
        yDistList.append(pts1[i][1] - pts2[j][1])
        distList.append(coordDist)
        result.append((i, j, minDist, coordDist))
        dMatchList.append(cv2.DMatch(i,j, minDist))
        curr.sort()
        superlist.append(curr)

        # Remove false mathing pairs by checking their distance
        if(curr[0]>100 or curr[0] > 0.6*curr[1]):
            result.pop()
            superlist.pop()
            dMatchList.pop()
            distList.pop()
            xDistList.pop()
            yDistList.pop()

    # Check if there are min 4 pairs remain, if not, return None
    # A min of 4 pairs of matches are needed to calculate transform
    if(len(result) < 4): return None, None, None

    # Cross check the average distance between two pictures
    # Translate in X, Y and resultant translate
    ave=sum(distList)/len(distList)
    aveX=sum(xDistList)/len(xDistList)
    aveY=sum(yDistList)/len(yDistList)
    print("avex= ", aveX, " avey= ", aveY, " ave=", ave)

    # Removing matching pairs which have a matching distance varies more than 10% of ave
    i=0
    while(i<len(distList)):
        if(abs(distList[i]-ave)/ave >0.9 ):
            result.pop(i)
            superlist.pop(i)
            dMatchList.pop(i)
            distList.pop(i)
            xDistList.pop(i)
            yDistList.pop(i)
            i = i-1
        i = i+1
    return result, superlist, dMatchList

def findHMatrix(dMatchList, kp1, kp2):
    # RANSAC algorithm to find the homography transform
    pts1 = np.asarray([[p.pt[0],p.pt[1]] for p in kp1 ])
    pts2 = np.asarray([[p.pt[0],p.pt[1]] for p in kp2 ])

    # Iterate over all possible transforms
    numOfPairs = len(dMatchList)
    bestCounter=0
    Hbest = np.zeros((3,3))

    # Trying iteration for 600 times (empirical number, canbe changed)
    for i in range(600):
        lotted=[]
        PointsL=[]
        PointsR=[]
        while(len(lotted) <4 ):
            dice=random.randint(0,numOfPairs-1)
            if not dice in lotted :
                lotted.append(dice)
        for pickedIndex in lotted:
            PointsL.append(pts1[dMatchList[pickedIndex].queryIdx]) #queryIdx
            PointsR.append(pts2[dMatchList[pickedIndex].trainIdx]) #trainIdx
        PointsL = np.array(PointsL, dtype = np.float32)
        PointsR = np.array(PointsR, dtype = np.float32)
        # current h transform matrix based on current picked 4 points
        HCurr = cv2.getPerspectiveTransform(PointsL, PointsR)

        # Evaluate the transfrom hMatrix
        # Calculate how many matching pairs agree with this hMatrix
        counter=0
        for dmatch in dMatchList:
            pt1 = pts1[dmatch.queryIdx]
            pt2 = pts2[dmatch.trainIdx]
            pt1_np = np.array([[pt1[0]],[pt1[1]],[1]])
            pt2_from1Mapped = np.dot(HCurr, pt1_np)
            if(abs(pt2[0]-pt2_from1Mapped[0,0]/pt2_from1Mapped[2,0])<0.8
               and abs(pt2[1]-pt2_from1Mapped[1,0]/pt2_from1Mapped[2,0])<0.8):
                counter = counter+1
        if(counter > bestCounter):
            bestCounter = counter
            Hbest = HCurr

    print("Find H: ")
    print(Hbest)
    print("total pairs: ", len(dMatchList))
    print("max matched: ", bestCounter)
    return Hbest


def mappedOrigin(img, homography):
    h,w,rgb = img.shape
    # calculate the converted corner coordinates
    corner_curr = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    corner_transformed = np.dot(homography, corner_curr)
    horizon_row = corner_transformed[0]/corner_transformed[2]
    vertical_row = corner_transformed[1]/corner_transformed[2]
    w_min = min(horizon_row)
    w_max = max(horizon_row)
    h_min = min(vertical_row)
    h_max = max(vertical_row)
    new_w = int(w_max-w_min)+1
    new_h = int(h_max-h_min)+1
    return w_min, h_min

def mappedSize(img, homography):
    h,w,rgb = img.shape
    #calculate the converted corner coordinates
    corner_curr = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    corner_transformed = np.dot(homography, corner_curr)
    horizon_row = corner_transformed[0]/corner_transformed[2]
    vertical_row = corner_transformed[1]/corner_transformed[2]
    w_min = min(horizon_row)
    w_max = max(horizon_row)
    h_min = min(vertical_row)
    h_max = max(vertical_row)
    new_w = int(w_max-w_min)+1
    new_h = int(h_max-h_min)+1
    dim = (new_w, new_h)
    w_ratio, h_ratio = max(new_w, w)/min(new_w,w), max(new_h, h)/min(new_h,h)
    return w_ratio, h_ratio

def sortImg(kps_descriptors, allImages):
    # Estabilish the order of the allImages
    img_order=[]
    leftOrUpMember=[]
    leftOrUpMemberMatchPairs=[]
    rightOrDownMember=[]
    rightOrDownMemberMatchPairs=[]

    # For each image, record all possible neighbors at two sides
    for elm in allImages:
        leftOrUpMember.append(list())
        leftOrUpMemberMatchPairs.append(list())
        rightOrDownMember.append(list())
        rightOrDownMemberMatchPairs.append(list())
    for i in range(len(kps_descriptors)-1):
        kp1,dscrptor1 = kps_descriptors[i]
        for j in range(i+1, len(kps_descriptors)):
            kp2, dscrptor2 = kps_descriptors[j]
            pairs, superlist, DmatchList1=findMatchPair(dscrptor1, dscrptor2, kp1, kp2)
            print("now comaring ", i, " and ", j, " pic" )
            #check if match exist
            if(DmatchList1 != None):
                hMatrix = findHMatrix(DmatchList1, kp1,kp2)
                img_match = cv2.drawMatches(allImages[i], kp1, allImages[j],
                kp2, DmatchList1, np.array([]))
                # A qucik check to see the matching
                show_image(img_match)
                new_w, new_h = mappedOrigin(allImages[i], hMatrix)
                print("Aligned origin = ", new_w, " , ", new_h, " im1_size:" ,
                                allImages[i].shape[1],",",allImages[i].shape[0],
                                " im2_size: " , allImages[j].shape[1], ",",allImages[j].shape[0])
                # Discard the match if the matching distance is out of bound
                if( ( ( new_w>0 and abs(new_w) >= 1.2*allImages[j].shape[1]) or
                    ( new_w<0 and abs(new_w) >= 1.2*allImages[i].shape[1] )  ) or
                    ( ( new_h>0 and abs(new_h) >= 1.2*allImages[j].shape[0]) or
                    ( new_h<0 and abs(new_h) >= 1.2*allImages[i].shape[0] )  ) ):
                    print("distance not valid"); continue
                # Calculate the image size which to be transformed
                new_w_r, new_h_r = mappedSize(allImages[i], hMatrix)
                print("Aligned scale = ", new_w_r, " , ", new_h_r)
                # Discard the match if the transformed size is way off comparing to original one
                if(max(new_w_r,new_h_r) >2): print("max scale change reached, H invalid"); continue
                show_image(img_match)

                # Add the found neighbors to the corresponding list
                if ( ((abs(new_w) > abs(new_h)) and (new_w<0)) or
                    ((abs(new_w) < abs(new_h)) and (new_h<0)) ):
                    rightOrDownMember[i].append(j)
                    rightOrDownMemberMatchPairs[i].append(len(DmatchList1))
                    leftOrUpMember[j].append(i)
                    leftOrUpMemberMatchPairs[j].append(len(DmatchList1))
                elif ( ((abs(new_w) > abs(new_h)) and (new_w>0)) or
                    ((abs(new_w) < abs(new_h)) and (new_h>0)) ):
                    leftOrUpMember[i].append(j)
                    leftOrUpMemberMatchPairs[i].append(len(DmatchList1))
                    rightOrDownMember[j].append(i)
                    rightOrDownMemberMatchPairs[j].append(len(DmatchList1))
            else:
                print("No relation Found.")


    # Analysis image order
    # For images with multiple neighbors, find the closest one as its adjacent image
    orderDict=dict()
    print("analysis pics order:")
    for i, elm in enumerate(allImages):
        # Analysis the neighbor on right or down side
        print("now analysis pic:  ", i)
        print("right adjacent is: ")
        print(rightOrDownMember[i])
        print(rightOrDownMemberMatchPairs[i])
        if (len(rightOrDownMember[i])>0):
            j=rightOrDownMember[i][np.argmax(rightOrDownMemberMatchPairs[i])]
            orderDict.update({i:j})
            print("estabilished ", i, " << ", j)
        # Analysis the neighbor on left or upper side
        print("left adjacent is: ")
        print(leftOrUpMember[i])
        print(leftOrUpMemberMatchPairs[i])
        if (len(leftOrUpMember[i])>0):
            j=leftOrUpMember[i][np.argmax(leftOrUpMemberMatchPairs[i])]
            orderDict.update({j:i})
            print("estabilished ", j, " << ", i)

    # Organize the tuple order data
    sortCounterL=[0 for i in range(len(allImages))]
    sortCounterR=[0 for i in range(len(allImages))]
    for i, j in orderDict.items():
        sortCounterL[j]+=1
        sortCounterR[i]+=1

    startIndex = np.argmin(sortCounterL)
    i=startIndex
    img_order.append(i)
    while( i in orderDict):
        i = orderDict.get(i)
        img_order.append(i)

    print(img_order)
    return img_order

def warpImg(img, homography):
    h,w,rgb = img.shape
    #calculate the converted corner coordinates
    corner_curr = np.array([[0, w, 0, w],
                            [0, 0, h, h],
                            [1, 1, 1, 1]])
    corner_transformed = np.dot(homography, corner_curr)
    horizon_row = corner_transformed[0]/corner_transformed[2]
    vertical_row = corner_transformed[1]/corner_transformed[2]
    w_min = min(horizon_row)
    w_max = max(horizon_row)
    h_min = min(vertical_row)
    h_max = max(vertical_row)
    new_w = int(w_max-w_min)+1
    new_h = int(h_max-h_min)+1
    dim = (new_w, new_h)
    orig_x = horizon_row[0]
    orig_y = vertical_row[0]

    # Compensate the offset
    offsetComp = np.array([[1, 0, -1*w_min],
                           [0, 1, -1*h_min],
                           [0, 0, 1]])
    homography=np.dot(offsetComp,homography)
    # Output the result
    warpedImg = cv2.warpPerspective(src=img, M=homography, dsize=dim)
    return warpedImg, (w_min, h_min)

#%%
def mosaicImg(img1, img2, featherBlending):
    # Mosaicing images
    # img2 is dest, position unchanged
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create(500)
    # The following parameters are empirical
    sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=10, sigma=1.5, contrastThreshold=0.12)
    kp1, dscrptor1 = sift.detectAndCompute(img1_grey, None)
    kp2, dscrptor2 = sift.detectAndCompute(img2_grey, None)
    pairs, superlist, DmatchList1=findMatchPair(dscrptor1, dscrptor2, kp1, kp2)
    hMatrix = findHMatrix(DmatchList1, kp1,kp2)

    # Drawing the matches
    imgXXX = cv2.drawMatches(img1, kp1, img2, kp2, DmatchList1, np.array([]))
    show_image(imgXXX)

    # Warping images
    img1, original = warpImg(img1, hMatrix)
    w_min, h_min = original
    print("Aligned Move = ", round(w_min,2), " ,", round(h_min ,2))
    w_min, h_min = int(w_min), int(h_min)
    h1,w1,rgb1 = img1.shape
    h2,w2,rgb2 = img2.shape
    # The combined pic size
    width = max(w1 + w_min, w2) - min(w_min, 0) #+ 1
    height = max(h1 + h_min, h2) - min(h_min, 0) #+ 1
    mosaic1 = np.zeros((height, width, 3), np.uint8)
    mosaic2 = np.zeros((height, width, 3), np.uint8)

    # Template for final output
    mosaic = np.zeros((height, width, 3), np.uint8)

    # Output the transformed coordinates data for check
    im1_org_H = max(0, h_min)
    im1_org_W = max(0, w_min)
    im2_org_H = max(0, -1*h_min)
    im2_org_W = max(0, -1*w_min)

    print("im1 shape =", w1,",",h1)
    print("im2 shape =", w2,",",h2)
    print("final shape = ", ",",  height, ",", width)

    print("im1's loc:")
    print(img1.shape)
    print(im1_org_H,",", im1_org_H +  h1, ",", im1_org_W, ",", im1_org_W +  w1 )

    print("im2's loc:")
    print(im2_org_H,",", im2_org_H +  h2, ",", im2_org_W, ",", im2_org_W +  w2)
    print(img2.shape)

    # if not feathering , just overlap the two pictures after warping
    if not featherBlending:
        mosaic[ im1_org_H : im1_org_H + h1  , im1_org_W : im1_org_W +  w1  , :] = img1
        mosaic[ im2_org_H : im2_org_H + h2  , im2_org_W : im2_org_W +  w2  , :] = img2
    else:
        mosaic1[ im1_org_H : im1_org_H + h1  , im1_org_W : im1_org_W +  w1  , :] = img1
        mosaic2[ im2_org_H : im2_org_H + h2  , im2_org_W : im2_org_W +  w2  , :] = img2
        #blending them together
        overlap_w = min(im2_org_W+w2, im1_org_W+w1) - max(im2_org_W, im1_org_W)
        overlap_H = min(im2_org_H+h2, im1_org_H+h1) - max(im2_org_H, im1_org_H)
        print("overlap width and height", overlap_w, ",",overlap_H )
        overlapW = min(overlap_w, overlap_w) # use 25 pixel
        overlapH = min(overlap_H, overlap_H)
        zeros = np.array([0,0,0])
        startPixelW = max(im1_org_W, im2_org_W)
        startPixelH = max(im1_org_H, im2_org_H)
        bendingCasePrint=True
        for row in range(height):
            for col in range(width):
                if(not np.array_equal(mosaic1[row, col, :], zeros)) and ( np.array_equal(mosaic2[row, col, :], zeros) ):
                    mosaic[row, col, :] = mosaic1[row, col, :]
                elif(np.array_equal(mosaic1[row, col, :], zeros)) and (not np.array_equal(mosaic2[row, col, :], zeros) ):
                    mosaic[row, col, :] = mosaic2[row, col, :]
                elif(not np.array_equal(mosaic1[row, col, :], zeros)) and (not np.array_equal(mosaic2[row, col, :], zeros) ):
                    if abs(h_min) < abs(w_min) :
                        if(bendingCasePrint): print("Horizontal Blending..")
                        if(w_min<0): weight = 1-weightFunction(col,startPixelW,overlapW)
                        if(w_min>0): weight = weightFunction(col,startPixelW,overlapW)
                    else:
                        if(bendingCasePrint): print("Vertical Blending..")
                        if(h_min<0): weight = 1-weightFunction(row, startPixelH,overlapH)
                        if(h_min>0): weight = weightFunction(row, startPixelH,overlapH)
                    bendingCasePrint=False
                    weight = weight
                    mosaic[row, col, :] = (weight)*mosaic1[row, col, :] + (1-weight)*mosaic2[row, col, :]
    return mosaic

#%%

def weightFunction(currIndex, startPos, overLap):
    # Feathering two images, each pixel value will come from the weighted sum
    # of two input
    blendingWidth=min(overLap, 25)
    weight =0.0
    if currIndex < int(startPos + (overLap - blendingWidth )/2 ):
        weight=0.0
    elif currIndex > int(startPos + (overLap + blendingWidth )/2 ):
        weight=1.0
    else:
        weight=  (currIndex - 0.5*(overLap - blendingWidth) ) / blendingWidth
        weight = max(min(1, weight),0)
    return weight



def mosaicAllImg(img_order, allImages, featherBlending):
    middleImgIndex = int((len(img_order))/2)
    print("middleImgIndex: ", middleImgIndex)

    # Merge left or upper
    counter=0
    i=1
    mosaic = allImages[img_order[0]]
    show_image(mosaic)
    while (i<=middleImgIndex):
        img2 = allImages[img_order[i]]
        print("processing: ", i-1, ",", i)
        show_image(img2)
        mosaic = mosaicImg(mosaic, img2, featherBlending)
        counter+=1
        # cv2.imwrite(img_path + "panorama_"+ str(counter) +".jpg", mosaic)
        show_image(mosaic)
        i=i+1
    print("finished left")
    allImages[img_order[middleImgIndex]] = mosaic

    # Merge right or downl
    j=len(img_order)-2
    mosaic = allImages[img_order[len(img_order)-1]]
    show_image(mosaic)

    while (j>=middleImgIndex):
        img2 = allImages[img_order[j]]
        print("processing ", j+1,",",j)
        show_image(img2)
        mosaic = mosaicImg(mosaic, img2, featherBlending)
        counter+=1
        # cv2.imwrite(img_path + "panorama_"+ str(counter) +".jpg", mosaic)
        show_image(mosaic)
        j=j-1
    print("finished right")

    return mosaic

#%%
#=====================main================================================
#=========================================================================

# Switch of whether feather blending
featherBlending=False # can be disabled for quicker stitching

# Find all image files in the directory
if len(sys.argv) >=2 :
    img_path ="../" + sys.argv[1] + "/"
else:
    img_path = "../data/"
onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
allImages=[]
if len(onlyfiles)==0: print("No files in folder!");   exit()
for filename in onlyfiles:
    if filename=="panorama.jpg": continue
    allImages.append(cv2.imread(img_path + filename, 1))
    print(filename)
#%%

# check if duplicate files exist
for i in range(len(allImages)-1):
    img_i = allImages[i]
    j=i+1
    while (j <len(allImages)):
        img_j=allImages[j]
        if (np.array_equal(img_i, img_j)):
            allImages.pop(j)
            print("removed duplicate ", j)
            j=j-1
        j=j+1

# convert to grey image
allImages_grey=[]
for img in allImages:
    allImages_grey.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# Retrieve SIFT feature
# sift = cv2.xfeatures2d.SIFT_create(500)
sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=10, sigma=1.5,contrastThreshold=0.12)
kps_descriptors=[]
for i,grey_img in enumerate(allImages_grey):
    kp, descriptor = sift.detectAndCompute(grey_img, None)
    kps_descriptors.append((kp, descriptor))
    # Display keypoints
    img_desp = cv2.drawKeypoints(allImages[i], kp, outImage= np.array([]), color=(0,0,255), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Image ", i, " found keypoints: ", len(kp))
    show_image(img_desp)

#sort the images in the right order
img_order = sortImg(kps_descriptors, allImages)
for imgId in img_order:
    print(onlyfiles[imgId])

#combined the sorted pic
finalImg = mosaicAllImg(img_order, allImages, featherBlending)
show_image(finalImg)
cv2.imwrite(img_path + "panorama.jpg", finalImg)
