from skimage.exposure import is_low_contrast
import imutils
import cv2

def LightCheck(imagePath):
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur the image slightly and perform edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    # initialize the text and color to indicate that the input image
    # is *not* low contrast
    text = "Low contrast: No"
    color = (0, 255, 0)
    if is_low_contrast(gray):
        return False
    # otherwise, the image is *not* low contrast, so we can continue
    # processing it
    else:
        # find contours in the edge map and find the largest one,
        # which we'll assume is the outline of our color correction
        # card
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw the largest contour on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        return True