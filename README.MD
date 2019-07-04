# Team 12 - FaceSwap

## Programing Language

* Front-End
  - HTML
  - CSS
  - JavaScript
* Back-End
  - Python
  - Django

## Public API

- Bored api

  Let's find you something  to do !

  [link](http://www.boredapi.com/)

- Bing image search api

  Let's find some image satisfying requirements !

  [link](https://azure.microsoft.com/zh-cn/services/cognitive-services/bing-image-search-api/)

- Merge Face

  Let's swap people's face and have fun!

  [link](https://api-cn.faceplusplus.com/imagepp/v1/mergeface)

- Detect Face

  Let's detect the human face from image!

  [link](https://api-cn.faceplusplus.com/facepp/v3/detect)

## Getting Activity

- User sets requirements(accessibility, participants, price, type) in the web front-end
- Back-end gets requirements  by using  **POST/GET????**
- Using function boreapi() to get the activity which satisfy the requirements from "Bored api" 

## Getting Picture of Swap-Face

- Sarting a GET request to Bing-image-search-api to get activity image by the keyword of activity from bored-api
- Asking user to upload one image with their front face inside
- Using the detect-face-api to detect face from both images
- Using the merge-face-api to swap the face form activity image to user image