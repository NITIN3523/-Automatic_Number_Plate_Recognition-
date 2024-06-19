import cv2
import Util
import os
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import tensorflow as tf

## deifne constants ##
class_name_path = 'model/classes.names'
weights_path = 'model/weights/model.weights'
cfgs_path = 'model/cfgs/darknet-yolov3.cfg'

## load class names ##
with open(class_name_path,'r') as f:        
    classnames = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

## load model ##
    net = cv2.dnn.readNetFromDarknet(cfgs_path,weights_path)
textdetector = easyocr.Reader(['en'])

## load image ##
imagespath = 'Input'
for file in os.listdir(imagespath):
    img_path = os.path.join(imagespath,file)
    img = cv2.imread(img_path)
    H,W,_ = img.shape

    ## convert image ##
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), True)

    ## get detections ##
    net.setInput(blob)
    detections = Util.get_outputs(net)

    ## bboxes, class_ids, confidence ##
    bboxes = []
    classIDes = []
    scores = []

    for detection in detections:
        bbox = detection[:4]
        x,y,w,h = bbox
        bbox = [int(x*W),int(y*H),int(w*W),int(h*H)]
        classID = np.argmax(detection[5:])
        score = np.amax(detection[5:])
        bboxes.append(bbox)
        classIDes.append(classID)
        scores.append(score)

    # print(bboxes)
    # print(classID)
    # print(score)

    # for bbox in bboxes:
    #     xc,yc,w,h = bbox
    #     cv2.rectangle(img, (xc - int(w / 2), yc - int(h / 2)),
    #                         (xc + int(w / 2), yc + int(h / 2)),(0,255,0),5)
        

    
    ## apply nms(non maximum separation) ##
    bboxes,classIDes,scores = Util.NMS(bboxes,classIDes,scores)

    # print(len(bboxes))
    # print(len(scores))
    # print(len(classIDes))
    # print(bboxes)
    # print(scores)
    # print(classIDes)
    # print(classnames)



    ## plot ##
    for bbox_, bbox in enumerate(bboxes):
        xc,yc,w,h = bbox
        # print(classnames[classIDes[bbox_]])
        # cv2.putText(img,classnames[classIDes[bbox_]],(int(xc - (w / 2) + 10), int(yc + (h / 2) - 30)),cv2.FONT_HERSHEY_SIMPLEX,4,(255,0,255),10)
        
        licenseplate = img[int(yc - (h / 2)):int(yc + (h / 2)),int(xc - (w / 2)):int(xc + (w / 2)):].copy()
        licenseplategray = cv2.cvtColor(licenseplate,cv2.COLOR_BGR2GRAY)
        _ , licenseplate_thres = cv2.threshold(licenseplategray,64,255,cv2.THRESH_BINARY_INV)
        textinfo = textdetector.readtext(licenseplate_thres)
        for info in textinfo:
            bbox,text,score = info
            if score > 0.4:
                print(text)

        cv2.rectangle(img, (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))), (255,0,255), 10)
        
        # plt.figure()
        # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # plt.figure()
        # plt.imshow(cv2.cvtColor(licenseplate,cv2.COLOR_BGR2RGB))
        # plt.figure()
        # plt.imshow(cv2.cvtColor(licenseplategray,cv2.COLOR_BGR2RGB))
        # plt.figure()
        # plt.imshow(cv2.cvtColor(licenseplate_thres,cv2.COLOR_BGR2RGB))
        # plt.show()
        
        img = cv2.cvtColor(licenseplate,cv2.COLOR_BGR2RGB)
        cv2.imwrite('Output/{}.png'.format(file),img)


# cv2.imshow('re',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)



