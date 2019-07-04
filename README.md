# Panorama-Image-Stittching-Detailed-Steps



## This python script can automatically estabilish the order of all images in one folder and stitch them according to the estabilished order.



### Find a set of correspondence 

​		a. Identify interest points/key points in both images

​        b. Summarize and estabilish their matches between feature

​        c. Recommend to use cv2.sift to compute interest points and descriptors  

​        d. (Recommend to use descriptors matchers from openCV

​    

### Fitting the homograph

 		a. Find the best homograph mapping

​		b. Recommend to refer to cv2.findhomegraphy (least square best fit)  

​        c. The package also offers robustified RANSAC or least-median (need to tweak the parameters to get best results)

​        d. The code provided here has manually implemented the RANSAC algorithm to find the best-fit homograph

  

### Image warping

​		a. Cv.warpPerspective

​        b. Need to warp the images according the homographys

  

### Image mosaicing

​        a. Blend

​        b. Alpha blending?

​				i. feathering

​            	ii. Pyramid blending

​            	iii. Multi-band blending# Panorama-Image-Stittching-Detailed-Steps