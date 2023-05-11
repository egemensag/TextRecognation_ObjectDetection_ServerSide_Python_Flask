import cv2

import lightDetection as l
import Preprocessing as p
import ExtractText as e




def textRecognition(image):
    light = l.LightCheck(image)

    if light:
        image = p.preprocessing(image)
        check, text = e.GetText(image)


        if check:
            return text
        else:
            return 'No text'

    else:
        return 'NO LIGHT'
