from ctypes import c_bool

import cv2
import numpy as np


class BoxIdentify:
    
    thresh = 80
    
    def __init__(self,thresh):
       self.thresh = thresh
    
    ft = 0

    def getContours(self, img, imgCoun, realimg):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        new = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            new.append([area])

        # print(new)
        maxi = max(new)[0]
        # print(maxi)
        # print(len(new))

        if maxi > 10000:

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area > 10000:
                    # print("hii")
                    cv2.drawContours(img, cnt, -1, (255, 0, 0), 1)
                    peri = cv2.arcLength(cnt, True)
                    # print(peri)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    # print(approx)
                    screencnt = approx
                    corner = len(approx)
                    x, y, w, h = cv2.boundingRect(approx)
                    # print(x, y, w, h)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    if corner == 4:
                        aspectratio = w / float(h)
                        #global ft
                        if self.ft == 0:
                            self.ft = 1
                            #global thresh
                            #self.thresh = self.thresh
                            self.thresh = self.thresh + 100
                            self.SudokuIdentifier()
                            exit()

                        else:

                            if 0.5 <= aspectratio <= 1.5:
                                objtype = "sudoku_box found"
                                print(objtype)
                                ratio = 1
                                warped = self.four_point_transform(imgCoun, screencnt.reshape(4, 2) * ratio)
                                warped = cv2.resize(warped, (300, 300))
                                cv2.imshow("real black and white", warped)
                                warped1 = self.four_point_transform(realimg, screencnt.reshape(4, 2) * ratio)
                                warped1 = cv2.resize(warped1, (300, 300))
                                #cv2.imshow("real", warped1)
                                cv2.imwrite('./utilis/Sodokuboxbw.png', warped)
                                cv2.waitKey(0)

                            else:
                                objtype = "sudoku_box found"
                                print(objtype)
                                warped = img[y:y + h, x:w + x]
                                warped = cv2.resize(warped, (300, 300))
                                warped1 = realimg[y:y + h, x:w + x]
                                warped1 = cv2.resize(warped1, (300, 300))
                                #cv2.imshow("real sodoku", warped1)
                                cv2.imwrite('./utilis/Sodokuboxbw.png', warped)
                                cv2.imshow("bw sodoku", warped)
                                cv2.waitKey(0)
                                # print(len(contours))

                    elif corner != 4:
                        self.thresh = self.thresh + 1
                        self.SudokuIdentifier()
                        exit()

                """else:
                    #global thresh
                    thresh = thresh + 1
                    SudokuIdentifier()
                    exit()"""

        elif len(new) < 1 or maxi < 10000:
            self.thresh += 1
            self.SudokuIdentifier()
            exit()

        else:
            print("no sudoku box found")
            
            

    # function required
    def order_points(self,pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect


    # function that converts any four points to square
    def four_point_transform(self,image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped



    def SudokuIdentifier(self):
        
        # read image
        img_grey = cv2.imread('./utilis/Sodoku.PNG', cv2.IMREAD_GRAYSCALE)


        # assign blue channel to zeros , convert to binary image
        img_binary = cv2.threshold(img_grey, self.thresh, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('./utilis/imagebw.jpg', img_binary)
        
        
        #resize original image
        img = cv2.imread("./utilis/imagebw.jpg")
        img = cv2.resize(img, (600, 550))
        imgCoun = img.copy()
        
        
        #resize binary image
        realimg = cv2.imread("./utilis/Sodoku.PNG")
        realimg = cv2.resize(realimg, (600, 550))
        
        #canny finds the edges from thresholds 
        realcanny = cv2.Canny(realimg, 40, 40)
        #cv2.imshow("imgcanny",realcanny)
        #cv2.waitKey(0)

        mask = cv2.Canny(img, 40, 40)
        #cv2.imshow("imgcannybin",mask)
        #cv2.waitKey(0)
        
        self.getContours(mask, imgCoun, realimg)
        
        

