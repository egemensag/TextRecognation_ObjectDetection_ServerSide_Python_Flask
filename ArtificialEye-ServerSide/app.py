import numpy as np
import argparse
import time
import cv2
import os

def fonksiyon(imagePath):
    detected_classes = []
    stringReturn =""
    labelsPath = 'obj.names'
    configPath = 'yolov3_custom.cfg'
    weightsPath = 'yolov3_custom_last.weights'
    image = imagePath
    c = 0.1
    t = 0.3
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    image = cv2.imread(image)
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > c:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, c, t)
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            #(x, y) = (boxes[i][0], boxes[i][1])
            #(w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            #color = [int(c) for c in COLORS[classIDs[i]]]
            #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            stringReturn += "\n" + text
            #print(text)
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        #0.5, color, 2)

    # show the output image
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    aloha = "".join(c for c in stringReturn if c.isalpha() or c in {'\n', ' '})
    return aloha