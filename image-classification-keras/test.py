# import pytesseract
# from PIL import Image, ImageEnhance, ImageFilter
#
# im = Image.open("examples/test.jpg") # the second one
# im = im.filter(ImageFilter.MedianFilter())
# enhancer = ImageEnhance.Contrast(im)
# im = enhancer.enhance(2)
# im = im.convert('1')
# im.save('temp2.jpg')
# text = pytesseract.image_to_string(Image.open('temp2.jpg'))
# print(text)

# import cv2
# import sys
# import pytesseract
#
# if __name__ == '__main__':
#
#     if len(sys.argv) < 2:
#         print('Usage: python ocr_simple.py image.jpg')
#         sys.exit(1)
#
#     # Read image path from command line
#     imPath = sys.argv[1]
#
#     # Uncomment the line below to provide path to tesseract manually
#     # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
#
#     # Define config parameters.
#     # '-l eng'  for using the English language
#     # '--oem 1' for using LSTM OCR Engine
#     config = ('-l eng --oem 1 --psm 3')
#
#     # Read image from disk
#     im = cv2.imread(imPath, cv2.IMREAD_COLOR)
#
#     # Run tesseract OCR on image
#     text = pytesseract.image_to_string(im, config=config)
#
#     # Print recognized text
#     print(text)

# !/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''
#
# import sys
# import cv2
#
# if __name__ == '__main__':
#     # If image path and f/q is not passed as command
#     # line arguments, quit and display help message
#     if len(sys.argv) < 3:
#         print(__doc__)
#         sys.exit(1)
#
#     # speed-up using multithreads
#     cv2.setUseOptimized(True);
#     cv2.setNumThreads(4);
#
#     # read image
#     im = cv2.imread(sys.argv[1])
#     # resize image
#     newHeight = 200
#     newWidth = int(im.shape[1] * 200 / im.shape[0])
#     im = cv2.resize(im, (newWidth, newHeight))
#
#     # create Selective Search Segmentation Object using default parameters
#     ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#
#     # set input image on which we will run segmentation
#     ss.setBaseImage(im)
#
#     # Switch to fast but low recall Selective Search method
#     if (sys.argv[2] == 'f'):
#         ss.switchToSelectiveSearchFast()
#
#     # Switch to high recall but slow Selective Search method
#     elif (sys.argv[2] == 'q'):
#         ss.switchToSelectiveSearchQuality()
#     # if argument is neither f nor q print help message
#     else:
#         print(__doc__)
#         sys.exit(1)
#
#     # run selective search segmentation on input image
#     rects = ss.process()
#     print('Total Number of Region Proposals: {}'.format(len(rects)))
#
#     # number of region proposals to show
#     numShowRects = 5
#     # increment to increase/decrease total number
#     # of reason proposals to be shown
#     increment = 50
#
#     while True:
#         # create a copy of original image
#         imOut = im.copy()
#
#         # itereate over all the region proposals
#         for i, rect in enumerate(rects):
#             # draw rectangle for region proposal till numShowRects
#             if (i < numShowRects):
#                 x, y, w, h = rect
#                 cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
#             else:
#                 break
#
#         # show output
#         cv2.imshow("Output", imOut)
#
#         # record key press
#         k = cv2.waitKey(0) & 0xFF
#
#         # m is pressed
#         if k == 109:
#             # increase total number of rectangles to show by increment
#             numShowRects += increment
#         # l is pressed
#         elif k == 108 and numShowRects > increment:
#             # decrease total number of rectangles to show by increment
#             numShowRects -= increment
#         # q is pressed
#         elif k == 113:
#             break
#     # close image show window
#     cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img_rgb = cv2.imread('examples/fiha2.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('capture.png',0)
# w, h = template.shape[::-1]
#
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#
# cv2.imwrite('res.png',img_rgb)

import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model

from keras_preprocessing import image

img_to_array = image.img_to_array

def sign_or_not(img):
    image = cv2.imread(img)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model('mymodel.model')

    (notsign, sign) = model.predict(image)[0]

    # build the label
    label = "parking sign" if sign > notsign else "No park sign"
    proba = sign if sign > notsign else notsign
    label = "{}: {:.2f}%".format(label, proba * 100)
    return  label
def detect_sign(img):

    #img_rgb = cv2.imread('examples/fiha2.jpg')
    img_rgb = cv2.imread(img)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = [cv2.imread('train/capture.png',0),cv2.imread('train/tom.png',0),
                cv2.imread('train/nopark2.png',0),cv2.imread('train/nopark4.png',0),
                cv2.imread('train/nopark5.png',0),cv2.imread('train/park1.png',0),
                cv2.imread('train/park2.png',0),cv2.imread('train/park3.png',0),
                cv2.imread('train/park4.png',0),cv2.imread('train/park5.png',0),
                cv2.imread('train/nopark6.png',0)
                ]
    for image in template :
        w, h = image.shape[::-1]
        res = cv2.matchTemplate(img_gray,image,cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.putText(img_rgb, sign_or_not('examples/fiha2.jpg'), (10, 25),  cv2.FONT_HERSHEY_DUPLEX,
        0.4, (140, 255, 100), 1)
    cv2.imwrite('resul-bk1.jpg',img_rgb)

#test image
detect_sign('examples/bk1.jpg')