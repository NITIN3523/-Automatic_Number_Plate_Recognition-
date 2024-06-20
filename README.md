
# Automatic_Number_Plate_Recognition
***util.py:*** This file serves as a utility module, encapsulating reusable functionalities for the application. Notably, it includes a Non-Maximum Suppression (NMS) function. NMS is a crucial technique in object detection tasks. It helps reduce redundancy by eliminating overlapping bounding boxes for the same object. This ensures each object is represented by a single bounding box with the highest confidence score, improving detection accuracy.

***main.py:*** This file acts as the program's core functionality. It leverages the utility functions provided by util.py. Here's a breakdown of its potential functionalities:

**1. Model Loading:** It loads the pre-trained object detection model along with its associated weights and configuration files (cnfs path).

**2. Object Detection:** The loaded model is used to detect objects of interest within the input frames or images.

**3. Optical Character Recognition (OCR):** The script potentially integrates OCR functionality to recognize text on detected objects, likely focusing on license plates in this case.

**4. Output Management:** The script manages the output, potentially saving processed images with detected and labeled objects in a designated "Output" folder. 


***out.py:*** This file might be an optional component responsible for visualizing the results. It could potentially display the processed images with detected objects and any recognized text (e.g., license plate numbers) on the screen.

## Weights Link
https://drive.google.com/file/d/1vXjIoRWY0aIpYfhj3TnPUGdmJoHnWaOc/edit

Download it and put it in Model ==> Weights folder 

## Output
![Screenshot 2024-06-20 134438](https://github.com/NITIN3523/Automatic_Number_Plate_Recognition-/assets/84659412/979be847-d599-4987-8689-59db823f680f)


