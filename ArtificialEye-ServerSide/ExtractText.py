import pytesseract
from pytesseract import Output
import numpy as np
from textblob import TextBlob

def GetText(image):
    text = pytesseract.image_to_string(image, lang='eng+tur')

    if text != '':
        return True, text
    else:
        return False, ''

