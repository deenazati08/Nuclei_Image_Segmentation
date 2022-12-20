# Nuclei_Image_Segmentation
 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


[Description](https://github.com/deenazati08/Nuclei_Image_Segmentation#description) // [Data](https://github.com/deenazati08/Nuclei_Image_Segmentation#data) // [Result](https://github.com/deenazati08/Nuclei_Image_Segmentation#result) // [Discussion](https://github.com/deenazati08/Nuclei_Image_Segmentation#discussion) // [Credits](https://github.com/deenazati08/Nuclei_Image_Segmentation#discussion)

## Description

An algorithm that can detect nuclei automatically to speed up research on a variety of diseases such as cancer, heart disease, and rare disorders. Such a tool has the potential to significantly accelerate the development of cures, benefiting those suffering from a variety of health conditions such as COPD, Alzheimer's, diabetes, and even the common cold.

As a result, identifying cell nuclei is an important first step in many research studies because it allows researchers to analyse the DNA contained within the nucleus, which contains the genetic information that determines each cell's function. Researchers can examine how cells respond to different treatments and gain insights into the underlying biological processes at work by identifying cell nuclei. An automated AI model for identifying nuclei has the potential to streamline drug testing and shorten the time it takes for new drugs to reach the public.


## Data

~ Model used :

<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/208713132-a252f3b8-589c-4824-958d-f77d354d7b74.png" width="700" height="700">
</p>

~ Train images output :

<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/208712678-e8f734ee-bd5d-412c-863f-f160893edc07.png">
</p>

~ Train masks output :

<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/208715283-005fc1db-a880-400c-8ccb-cfde5121244a.png">
</p>

~ Test images output :

<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/208715110-b793df21-cfe1-414e-b646-e2f9d2dcb7c7.png">
</p>

~ Test masks output :

<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/208715476-f84b3c19-bf64-4dfa-bfd8-609008565f5f.png">
</p>

## Result

Accuracy and Loss Graph from TensorBoard :
<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/208717188-32f72516-f773-49bc-ac11-9bceb619d822.jpg" width, height="350, 350"><img src="https://user-images.githubusercontent.com/120104404/208717311-08f2806f-f754-40dd-bd2b-31b55025f143.jpg" width, height="350, 350">
</p>

## Discussion

From the Accuracy and Loss Graph we can see that this not a really good model as the graph were stop while in a premature learning. It can be improved by using more epoch to develop the model and also use more callbacks to prevent overfitting data. 

## Credits

- https://www.kaggle.com/competitions/data-science-bowl-2018/overview
- The model architecture were develop by using MobileNetV2
