import cv2
import numpy as np

def slicer():
        
    test_digits = cv2.imread("./utilis/Sodokuboxbw.png", cv2.IMREAD_GRAYSCALE)
    test_digits = cv2.resize(test_digits,(900,900))

    cells = []

    test_digits = np.vsplit(test_digits, 9)
    test_cells = []
    b = 1
    for d in test_digits:
        dc = np.hsplit(d,9)

        for de in dc:
            dec = cv2.resize(de,(28,28))
            cv2.imshow("ce", dec)

            cv2.waitKey(0)
            name = "B" + str(b)
            dec = dec[2:26, 2:26]
            dec = cv2.resize(dec,(28,28))

            #cv2.imshow("ce", dec)
            #cv2.waitKey(1)


            cv2.imwrite(f'Digits/{name}.png', dec)
            #imgidentifier(b)
            #dec = cv2.imread(f'img/{name}.png',cv2.IMREAD_GRAYSCALE)
            #cv2.imshow("fi",dec)
            #cv2.waitKey(0)
            #b += 1
            #dec = dec.flatten()
            #test_cells.append(dec)