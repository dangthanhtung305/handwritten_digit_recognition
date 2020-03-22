import sys
import cv2
import requests
import math
import xlrd
import xlwt
import numpy as np
import argparse as ap
from xlutils.copy import copy
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QTableWidgetItem
from PyQt5.uic import loadUi
from sklearn.externals import joblib
from skimage.feature import hog
from scipy import ndimage
from collections import Counter
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import itertools

# url = ('http://10.42.0.216:8080/shot.jpg')
# url = ('http://192.168.43.31:8080/shot.jpg') 

img_dis = []
class BangDiemGUI(QDialog):
    def __init__(self):
        super(BangDiemGUI, self).__init__()
        loadUi('BangDiemGUI.ui', self)
        self.camera_flag = 0
        self.originalImage = None
        self.originalCapImage = None
        self.predicted_digits_mssv = []
        self.predicted_digits_point = []
        self.predicted_digits_rubric = []
        self.MssvList = []
        self.ScoresList = []
        self.RubricList = []
        self.org_labels = [1, 5, 1, 2, 8, 6, 7, 8, 5, 1, 5, 1, 3, 9, 4, 7, 6, 0, 1, 5, 0, 6, 4, 3, 2, 9, 0, 1, 5, 1, 0, 2, 6, 7, 8, 5, 1, 5, 0, 2, 4, 8, 9, 4, 5]
        self.pred_labels = []
        self.class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9]
        self.urlEdit.setText('192.168.43.31:8080')  
        self.loadButton.clicked.connect(self.loadClick)
        self.processedButton.clicked.connect(self.processedClick)
        self.newFileButton.clicked.connect(self.createNewFile)
        self.addFileButton.clicked.connect(self.addFile)
        self.validateButton.clicked.connect(self.validateClick)
        self.tableWidget.setRowCount(1)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setItem(0, 0, QTableWidgetItem("MSSV"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem("Điểm Tổng"))
        self.tableWidget.setItem(0, 2, QTableWidgetItem("Điểm Rubric"))


    def loadClick(self):
        if self.camera_flag == 0:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(20)
            self.camera_flag = 1
        else:
            print('timer flag: ' + str(self.camera_flag))
            self.timer.stop()
            self.camera_flag = 0


    def getOuterPoints(self, rcCorners):
        ar = [];
        ar.append(rcCorners[0, 0, :]);
        ar.append(rcCorners[1, 0, :]);
        ar.append(rcCorners[2, 0, :]);
        ar.append(rcCorners[3, 0, :]);
        x_sum = sum(rcCorners[x, 0, 0] for x in range(len(rcCorners))) / len(rcCorners)
        y_sum = sum(rcCorners[x, 0, 1] for x in range(len(rcCorners))) / len(rcCorners)

        def algo(v):
            return (math.atan2(v[0] - x_sum, v[1] - y_sum)
                    + 2 * math.pi) % 2 * math.pi
            ar.sort(key=algo)

        return (ar[3], ar[0], ar[1], ar[2])


    def order_points(self, pts):
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


    def four_point_transform(self, image, pts):
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
        return (warped, maxWidth, maxHeight)


    def sort_contours(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)


    def is_contour_bad(self,c_2):
        flag = False
        if (cv2.contourArea(c_2)<30):
            flag = True
        return flag


    def update_frame(self):
        url = self.urlEdit.text()
        url = 'http://' + url + '/shot.jpg'
        img_resp1 = requests.get(url)
        img_arr1 = np.array(bytearray(img_resp1.content), dtype=np.uint8)
        imgOr = cv2.imdecode(img_arr1, 1)
        # cv2.imshow("img", imgOr)
        # imgOr = self.xoayanh(imgOr, 180)
        self.displayImage(imgOr, 1)


    def xoayanh(self, inputImg, goc):
        w = inputImg.shape[1]
        h = inputImg.shape[0]
        rangle = np.deg2rad(int(goc))
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * 1
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * 1
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), goc, 1)  # tam quay, goc, scale
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        dst = cv2.warpAffine(inputImg, rot_mat, (int(np.math.ceil(nw)), int(np.math.ceil(nh))))  # lam tron
        return dst


    def getBestShift(img):
        cy, cx = ndimage.measurements.center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)
        return (shiftx, shifty)


    def shift(img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted


    def digitRecognition(self):
        # url = self.urlEdit.text()
        # url = 'http://' + url + '/shot.jpg'
        # img_resp2 = requests.get(url)
        # img_arr2 = np.array(bytearray(img_resp2.content), dtype=np.uint8)
        # img = cv2.imdecode(img_arr2, 1)
        # #img = self.xoayanh(img, 180)
        # cv2.imwrite("write_img/org14.jpg",img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        err_detect_table = False
        err_detect_mssv = False
        err_detect_point = False

        img_err = cv2.imread('error.png')
        img = cv2.imread('./test_images/final3.jpg',0)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # cv2.imshow("blur image.jpg",blur)
        th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # cv2.imshow("adaptiveThreshold image.jpg",th2)
        img_bin = 255 - th2
        conn_kernel = np.ones((3, 3), np.uint8)
        conn = cv2.dilate(img_bin, conn_kernel, iterations=1)
        # cv2.imshow('con', conn)
        # self.displayImage(conn, 2)
        img_skew, contours0, hierarchy = cv2.findContours(
            conn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape[:2]
        image_candidates = img.copy()
        # img_coutours
        # cv2.drawContours(img, contours0, -1, (255,0,0), 1)
        # cv2.imshow('contours0', img)
        # biggest rectangle
        size_rectangle_max = 0
        # big_rectangle = [1,1]
        is_table = False
        big_rectangle = []
        for i in range(len(contours0)):
            # aproximate countours to polygons

            approximation = cv2.approxPolyDP(contours0[i], 250, True)
            # print("Chieu dai " +(str(i)) + " app la =" + str(len(approximation)))
            # has the polygon 4 sides?
            if (not (len(approximation) == 4)):

                # print("Choose i : " +str(i))
                continue
            # is the polygon convex ?
            if (not cv2.isContourConvex(approximation)):
                # print("Not Convet")
                continue
            # epsilon = 0.04*cv2.arcLength(contours0[i],True)
            # print("epsilon: " + str(epsilon))
            is_table = True
            # area of the polygon
            size_rectangle = cv2.contourArea(approximation)
            # store the biggest
            if size_rectangle > size_rectangle_max:
                size_rectangle_max = size_rectangle
                big_rectangle = approximation
                # print("Have big rec")
        # show the best candidate
        approximation = big_rectangle

        #Check
        if (is_table == False):
            self.displayImage(img_err, 2)
            err_detect_table = True
            return False
        # print("Choose : " + str(len(approximation)))
        # print approximation
        # for i in range(len(approximation)):
        for i in range(len(approximation)):
            cv2.line(image_candidates, (big_rectangle[(i % 4)][0][0], big_rectangle[(i % 4)][0][1]),
                     (big_rectangle[((i + 1) % 4)][0][0], big_rectangle[((i + 1) % 4)][0][1]),
                     (255, 0, 0), 2)
        # cv2.imshow('image_candidates',image_candidates)
        img_dis = image_candidates
        self.displayImage(image_candidates, 2)
        outerPoints = self.getOuterPoints(approximation)
        points2 = np.array(outerPoints, np.float32)
        x1 = int(points2[0, 0]) + int(0)
        x2 = int(points2[1, 0]) - int(0)
        x3 = int(points2[2, 0]) + int(0)
        x4 = int(points2[3, 0]) - int(0)
        y1 = int(points2[0, 1]) + int(0)
        y2 = int(points2[1, 1]) + int(0)
        y3 = int(points2[2, 1]) - int(0)
        y4 = int(points2[3, 1]) - int(0)

        max_x = int(max([points2[0, 0], points2[1, 0], points2[2, 0], points2[3, 0]]))
        min_x = int(min([points2[0, 0], points2[1, 0], points2[2, 0], points2[3, 0]]))
        max_y = int(max([points2[0, 1], points2[1, 1], points2[2, 1], points2[3, 1]]))
        min_y = int(min([points2[0, 1], points2[1, 1], points2[2, 1], points2[3, 1]]))
        cropped_w = max_x - min_x
        cropped_h = max_y - min_y
        cropped_org = img[min_y:max_y, min_x:max_x]
        # cv2.imshow('cropped_org',cropped_org)

        # cv2.circle(img, (x1, y1), 4, (0, 255, 0), 3)
        # cv2.circle(img, (x2, y2), 6, (0, 255, 0), 3)
        # cv2.circle(img, (x3, y3), 8, (0, 255, 0), 3)
        # cv2.circle(img, (x4, y4), 4, (0, 255, 0), 3)
        # cv2.imshow('img', img)
        pts = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

        # apply the four point tranform to obtain a "birds eye view" of
        # the image
        warped, max_W, max_H = self.four_point_transform(img_bin, pts)
        # print("W and H: " + str(max_W) + " and " + str(max_H))

        #check
        if ((max_W / max_H > 4) or (max_W / max_H < 2)):
            self.displayImage(img_err, 2)
            err_detect_table = True
            return False

        # cv2.imshow('warped',warped)
        warped_org, max_W, max_H = self.four_point_transform(img, pts)
        #cv2.imshow('warped_org', warped_org)

        # Load classifier
        #clf, pp = joblib.load("digit-recognition-edited/digits_cls_py3_test3.pkl")
        # clf, pp = joblib.load("../digit-recognition-edited/digits_cls_py3_L2.pkl")
        clf, pp = joblib.load("../Classifier/digits_cls_py3_L2_4x4_2x2.pkl")

        ######################################################
        # MSSV#
        ######################################################
        x_mssv = int((max_W * 12) / 17.5)
        y_mssv = int((max_H * 0.3) / 7)
        w_mssv = int((max_W * 5) / 17.5)
        h_mssv = int((max_H * 1) / 7)
        img_mssv = warped_org[y_mssv:y_mssv + h_mssv, x_mssv:x_mssv + w_mssv]
        #cv2.imshow('mssv.png', img_mssv)
        ret, img_mssv_thresh = cv2.threshold(img_mssv, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((1, 31), np.uint8)
        img_mssv_thresh = cv2.dilate(img_mssv_thresh, kernel, iterations=1)
        img_mssv_thresh = cv2.erode(img_mssv_thresh, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        img_mssv_thresh = cv2.dilate(img_mssv_thresh, kernel, iterations=1)
        #cv2.imshow('img_mssv_thresh.png', img_mssv_thresh)
        (_, contour_mssv, hierarchy) = cv2.findContours(img_mssv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areaMax = 0
        xpM = 0
        ypM = 0
        wpM = 0
        hpM = 0
        if len(contour_mssv) > 0:
            for cnt_m in range(len(contour_mssv)):
                xm, ym, wm, hm = cv2.boundingRect(contour_mssv[cnt_m])
                area = cv2.contourArea(contour_mssv[cnt_m])
                if area > areaMax:
                    areaMax = area
                    xpM = xm
                    ypM = ym
                    wpM = wm
                    hpM = hm
            y_adj = ypM - 2
            x_adj = xpM - 2
            if (y_adj<0):
                y_adj = 0
            if (x_adj<0):
                x_adj = 0
            mini_img_mssv = img_mssv[y_adj: y_adj + hpM+4, x_adj:x_adj + wpM+4]
            #cv2.imshow('mini_img_mssv', mini_img_mssv)
            #mini_img_mssv = cv2.copyMakeBorder(mini_img_mssv,10,10,10,10,cv2.BORDER_REPLICATE)
            thresh_mssv = cv2.adaptiveThreshold(mini_img_mssv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13,8)
            #cv2.imshow('thresh_mssv', thresh_mssv)
            mini_img_mssv, contours_mssv, hierarchy = cv2.findContours(thresh_mssv, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            (contours_mssv, boundingBoxes) = self.sort_contours(contours_mssv, method="left-to-right")
            # cv2.drawContours(mini_img_mssv, contours_mssv, -1, (128,128,128), 2)
            # cv2.imshow('contours_mssv', mini_img_mssv)
            mini_img_mssv = 255 - mini_img_mssv
            #cv2.imshow('mini_img_mssv111', mini_img_mssv)
            mask = np.ones(mini_img_mssv.shape[:2], np.uint8) * 255
            for c in contours_mssv:
                if self.is_contour_bad(c):
                    cv2.drawContours(mask, [c], -1, 0, -1)
            mini_img_mssv = cv2.bitwise_and(mini_img_mssv, mini_img_mssv, mask=mask)
            #cv2.imshow('mini_img_mssv_filter', mini_img_mssv)
            idx_mssv = 0
            # mssv_factor = 6
            # mssv = 0
            leng_past = 0
            predicted_digits_mssv_temp = []
            for c_mssv in contours_mssv:
                x, y, w, h = cv2.boundingRect(c_mssv)

                if w > 5 and h > 5 and w * h > 100 and w < wpM / 3 and x>=leng_past:
                    # print("area " +str(idx_mssv+1) +" "+ str(w*h))
                    # print("aa " + str(cv2.contourArea(c_mssv)))
                    idx_mssv = idx_mssv + 1
                    leng_past = x + w
                    #print("mssv " + str(idx_mssv) + " - x " + str(x) + " - w " + str(w) + " - w*h " + str(w * h))
                    img_digit = mini_img_mssv[y:y + h, x:x + w]
                    #img_digit = 255 - img_digit

                    #cv2.imshow("mssv_" + str(idx_mssv) + '.png', img_digit)
                    while np.sum(img_digit[0]) == 0:
                        img_digit = img_digit[1:]
                    while np.sum(img_digit[:,0]) == 0:
                        img_digit = np.delete(img_digit,0,1)
                    while np.sum(img_digit[-1]) == 0:
                        img_digit = img_digit[:-1]
                    while np.sum(img_digit[:,-1]) == 0:
                        img_digit = np.delete(img_digit,-1,1)
                    rows,cols = img_digit.shape
                    if rows > cols:
                        factor = 20.0/rows
                        rows = 20
                        cols = int(round(cols*factor))
                        img_digit = cv2.resize(img_digit, (cols,rows),interpolation = cv2.INTER_AREA)
                    else:
                        factor = 20.0/cols
                        cols = 20
                        rows = int(round(rows*factor))
                        img_digit = cv2.resize(img_digit, (cols, rows),interpolation = cv2.INTER_AREA) #sua 14/11/2018

                    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
                    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
                    img_digit = np.lib.pad(img_digit,(rowsPadding,colsPadding),'constant')
                    #(shiftx,shifty) = self.getBestShift(img_digit)
                    cy,cx = ndimage.measurements.center_of_mass(img_digit)
                    rows_new,cols_new = img_digit.shape
                    shiftx = np.round(cols_new/2.0-cx).astype(int)
                    shifty = np.round(rows_new/2.0-cy).astype(int)
                    #shifted = self.shift(img_digit,shiftx,shifty)
                    #rows,cols = img_digit.shape
                    M = np.float32([[1,0,shiftx],[0,1,shifty]])
                    shifted = cv2.warpAffine(img_digit,M,(cols_new,rows_new))

                    img_digit = shifted
                    # cv2.imshow("mssv_" + str(idx_mssv) + '.png', img_digit)
                    #test1: 8,8 - 2,2
                    #test2: 7,7 - 4,4    
                    #test3: 7,7 - 2,2
                    #test4: 4,4 - 2,2
                    roi_hog_fd = hog(img_digit, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), visualize=False, block_norm='L2')
                    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
                    nbr = clf.predict(roi_hog_fd)
                    
                    self.pred_labels.append(nbr[0])
                    predicted_digits_mssv_temp.append(nbr[0])
                    # mssv = mssv + pow(10,mssv_factor)*int(nbr[0])
                    # mssv_factor = mssv_factor - 1


            # print("MSSV: " + str(mssv))
            self.predicted_digits_mssv.append(predicted_digits_mssv_temp)
            #print(self.predicted_digits_mssv)
            #Check
            if (idx_mssv != 7):
                self.displayImage(img_err, 2)
                print("fail_mssv8")
                err_detect_mssv = True
                return False
        ######################################################
        #Point#
        ######################################################
        x_point = x_mssv
        y_point = int((max_H * 6.4) / 7)
        w_point = w_mssv
        h_point = max_H - y_point;
        #h_point = int((max_H * 0.53) / 7)
        img_point = warped_org[y_point:y_point + h_point, x_point:x_point + w_point]
        #cv2.imshow('point.png', img_point)
        thresh_point = cv2.adaptiveThreshold(img_point, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 8)
        kernel = np.ones((1, 31), np.uint8)
        thresh_point = cv2.dilate(thresh_point, kernel, iterations=1)
        thresh_point = cv2.erode(thresh_point, kernel, iterations=1)
        kernel = np.ones((5, 1), np.uint8)
        thresh_point = cv2.erode(thresh_point, kernel, iterations=1)
        thresh_point = cv2.dilate(thresh_point, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        thresh_point = cv2.dilate(thresh_point, kernel, iterations=1)
        kernel = np.ones((1, 17), np.uint8)
        thresh_point = cv2.dilate(thresh_point, kernel, iterations=1)
        (_, contours_point2, hierarchy) = cv2.findContours(thresh_point, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areaMax = 0
        xpM = 0
        ypM = 0
        wpM = 0
        hpM = 0
        # print(str(len(contours_point2)))
        if len(contours_point2) > 0:
            for cnt_p in range(len(contours_point2)):
                xp, yp, wp, hp = cv2.boundingRect(contours_point2[cnt_p])
                area = cv2.contourArea(contours_point2[cnt_p])
                if area > areaMax:
                    areaMax = area
                    xpM = xp
                    ypM = yp
                    wpM = wp
                    hpM = hp

            y_adj = ypM - 4
            x_adj = xpM - 4
            if (y_adj<0):
                y_adj = 0
            if (x_adj<0):
                x_adj = 0
            mini_img_point = img_point[y_adj:y_adj + hpM+8, x_adj: x_adj + wpM+8]
            #cv2.imshow('mini_img_point', mini_img_point)
            thresh_point = cv2.adaptiveThreshold(mini_img_point, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13,8)
            #cv2.imshow('thresh_point', thresh_point)
            mini_img_point, contours_point, hierarchy = cv2.findContours(thresh_point, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            (contours_point, boundingBoxes) = self.sort_contours(contours_point, method="left-to-right")
            # cv2.drawContours(mini_img_point, contours_point, -1, (128,128,128), 1)
            # cv2.imshow('contours_point', mini_img_point)
            mini_img_point = 255 - mini_img_point
            #cv2.imshow('mini_img_point111', mini_img_point)
            mask = np.ones(mini_img_point.shape[:2], np.uint8) * 255
            for c in contours_point:
                if self.is_contour_bad(c):
                    cv2.drawContours(mask, [c], -1, 0, -1)
            mini_img_point = cv2.bitwise_and(mini_img_point, mini_img_point, mask=mask)
            #cv2.imshow('mini_img_point_filter', mini_img_point)
            idx_point = 0
            # point_factor = 0
            # point = 0
            leng_past = 0
            predicted_digits_point_temp = []
            for c_point in contours_point:
                x, y, w, h = cv2.boundingRect(c_point)
                #print("area " + str(w*h))
                if w > 5 and h > 5 and w <= wpM  and w * h > 100 and leng_past>=0: #and w * h < 700 :
                    idx_point = idx_point + 1
                    leng_past = x + w
                    img_digit = mini_img_point[y:y + h, x:x + w]
                    #cv2.imshow("point_" + str(idx_point) + '.png', img_digit)
                    while np.sum(img_digit[0]) == 0:
                        img_digit = img_digit[1:]
                    while np.sum(img_digit[:,0]) == 0:
                        img_digit = np.delete(img_digit,0,1)
                    while np.sum(img_digit[-1]) == 0:
                        img_digit = img_digit[:-1]
                    while np.sum(img_digit[:,-1]) == 0:
                        img_digit = np.delete(img_digit,-1,1)
                    rows,cols = img_digit.shape
                    if rows > cols:
                        factor = 20.0/rows
                        rows = 20
                        cols = int(round(cols*factor))
                        img_digit = cv2.resize(img_digit, (cols,rows))
                    else:
                        factor = 20.0/cols
                        cols = 20
                        rows = int(round(rows*factor))
                        img_digit = cv2.resize(img_digit, (cols, rows))

                    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
                    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
                    img_digit = np.lib.pad(img_digit,(rowsPadding,colsPadding),'constant')

                    cy,cx = ndimage.measurements.center_of_mass(img_digit)
                    rows_new,cols_new = img_digit.shape
                    shiftx = np.round(cols_new/2.0-cx).astype(int)
                    shifty = np.round(rows_new/2.0-cy).astype(int)
                    #shifted = self.shift(img_digit,shiftx,shifty)
                    #rows,cols = img_digit.shape
                    M = np.float32([[1,0,shiftx],[0,1,shifty]])
                    shifted = cv2.warpAffine(img_digit,M,(cols_new,rows_new))

                    img_digit = shifted
                    # cv2.imshow("point_" + str(idx_point) + '.png', img_digit)
                    #test3: 7,7 - 2,2
                    #test4: 4,4 - 2,2
                    roi_hog_fd = hog(img_digit, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), visualize=False, block_norm='L2')
                    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
                    nbr = clf.predict(roi_hog_fd)
                    # print(str(nbr[0]))
                    if idx_point < 3:
                        # print('idx point: ' + str(idx_point))
                        self.pred_labels.append(nbr[0])
                        predicted_digits_point_temp.append(nbr[0])
                    # point = point + pow(10,point_factor)*int(nbr[0])
                    # point_factor = point_factor - 1

            # print("Point: " + str(point))
            self.predicted_digits_point.append(predicted_digits_point_temp)
            #print(self.predicted_digits_point)
            # cv2.imshow('mini_img_point', mini_img_point)
            #check
            if (idx_point==0 or  idx_point>3):
                self.displayImage(img_err, 2)
                err_detect_point = True
                return False

        ###########################################################
        # Rubric#
        ###########################################################
        area_max = [0, 0, 0, 0, 0]
        r1 = 1
        for i in range(0, 5):
            x_r1 = int((max_W * (4.5 + i * 1.5)) / 17.5)
            y_r1 = int((max_H * 5) / 7)
            w_r1 = int((max_W * 1) / 17.5)
            h_r1 = int((max_H * 0.4) / 7)
            img_r1 = warped_org[y_r1:y_r1 + h_r1, x_r1:x_r1 + w_r1]
            #cv2.imshow('img_r1_' +str(i)+'.png', img_r1)
            thresh_r1 = cv2.adaptiveThreshold(img_r1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
            img_bin_r1 = 255 - thresh_r1
            img_bin_r1, contours_r1, hierarchy = cv2.findContours(img_bin_r1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img, contours_r1, -1, (255,0,0), 1)
            # cv2.imshow('contours_r1',img_r1)
            j = 0
            for j in range(len(contours_r1)):
                area = cv2.contourArea(contours_r1[j])
                # print (area)
                if (area > area_max[i]):
                    area_max[i] = area
                    # print "max o " + str(i) +" " + str(area_max[i])
                    # print (area_max[i])

            if (i > 0) and (area_max[i] > area_max[0]):
                r1 = i + 1
                area_max[0] = area_max[i]
            #cv2.imshow('img_r1.png', img_r1)

        area_max = [0, 0, 0, 0, 0]
        r2 = 1
        for i in range(0, 5):
            x_r2 = int((max_W * (4.5 + i * 1.5)) / 17.5)
            y_r2 = int((max_H * 5.5) / 7)
            w_r2 = int((max_W * 1) / 17.5)
            h_r2 = int((max_H * 0.4) / 7)
            img_r2 = warped_org[y_r2:y_r2 + h_r2, x_r2:x_r2 + w_r2]
            #cv2.imshow('img_r2_' +str(i)+'.png', img_r2)
            thresh_r2 = cv2.adaptiveThreshold(img_r2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
            img_bin_r2 = 255 - thresh_r2
            img_bin_r2, contours_r2, hierarchy = cv2.findContours(img_bin_r2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            j = 0
            for j in range(len(contours_r2)):
                area = cv2.contourArea(contours_r2[j])
                # print area
                if (area > area_max[i]):
                    area_max[i] = area
                    # print "max o " + str(i) +" " + str(area_max[i])

            if (i > 0) and (area_max[i] > area_max[0]):
                r2 = i + 1
                area_max[0] = area_max[i]
            #cv2.imshow('img_r2.png', img_r2)

        area_max = [0, 0, 0, 0, 0]
        r3 = 1
        for i in range(0, 5):
            x_r3 = int((max_W * (4.5 + i * 1.5)) / 17.5)
            y_r3 = int((max_H * 6) / 7)
            w_r3 = int((max_W * 1) / 17.5)
            h_r3 = int((max_H * 0.4) / 7)
            img_r3 = warped_org[y_r3:y_r3 + h_r3, x_r3:x_r3 + w_r3]
            #cv2.imshow('img_r3_' +str(i)+'.png', img_r3)
            thresh_r3 = cv2.adaptiveThreshold(img_r3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
            img_bin_r3 = 255 - thresh_r3
            img_bin_r3, contours_r3, hierarchy = cv2.findContours(img_bin_r3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            j = 0
            for j in range(len(contours_r3)):
                area = cv2.contourArea(contours_r3[j])
                # print area
                if (area > area_max[i]):
                    area_max[i] = area
                    # print "max o " + str(i) +" " + str(area_max[i])

            if (i > 0) and (area_max[i] > area_max[0]):
                r3 = i + 1
                area_max[0] = area_max[i]
            #cv2.imshow('img_r3.png', img_r3)

        # print("Rubric point: " + str(r1) + " " + str(r2) + " " + str(r3))
        # rubric = r1 * 100 + r2 * 10 + r3
        self.predicted_digits_rubric.append([r1, r2, r3])


    def prepareData(self, mssv, point, rubric):
        ###########################################################
        # Excel#
        ###########################################################
        self.MssvList.append(mssv)
        self.ScoresList.append(point)
        self.RubricList.append(rubric)
        rangeList = len(self.MssvList)
        self.tableWidget.insertRow(self.tableWidget.rowCount());
        self.tableWidget.setItem(rangeList, 0, QTableWidgetItem(str(self.MssvList[rangeList - 1])))
        self.tableWidget.setItem(rangeList, 1, QTableWidgetItem(str(self.ScoresList[rangeList - 1])))
        self.tableWidget.setItem(rangeList, 2, QTableWidgetItem(str(self.RubricList[rangeList - 1])))


    def calculateNumber(self, predicted_digits, number_factor):
        if predicted_digits == []:
            pass
        else:
            col_len = len(predicted_digits[0])
            number = 0
            for index in range(col_len):
                digit_col = [col[index] for col in predicted_digits]
                # print("Digit point " + str(index) + ": " + str(max(digit_col, key=digit_col.count)))
                predicted_digit = (max(digit_col, key=digit_col.count))
                number = number + pow(10,number_factor)*int(predicted_digit)
                number_factor = number_factor - 1
            return number


    def plotConfusionMatrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def processedClick(self):
        self.predicted_digits_mssv = []
        self.predicted_digits_point = []
        self.predicted_digits_rubric = []
        # for num_of_times in range(5):
        self.digitRecognition()
        # test_array = [[1, 5, 1, 3, 9, 8, 0], 
        #               [4, 5, 1, 3, 9, 8, 0], 
        #               [1, 5, 1, 9, 9, 8, 0], 
        #               [1, 5, 1, 3, 9, 8, 0], 
        #               [1, 5, 1, 3, 9, 8, 0]]
        mssv_factor = 6
        point_factor = 0
        rubric_factor = 2
        # print("fssfh: ")
        # print(self.predicted_digits_point)
        # print("\nknhnoi: ")
        # print(self.predicted_digits_mssv)
        mssv = self.calculateNumber(self.predicted_digits_mssv,mssv_factor)
        point = self.calculateNumber(self.predicted_digits_point,point_factor)
        rubric = self.calculateNumber(self.predicted_digits_rubric,rubric_factor)

        self.prepareData(mssv, point, rubric)


    def validateClick(self):
        # self.org_labels = [0, 2, 1, 8, 9, 6, 2, 3, 4, 5, 8, 7, 7]
        # self.pred_labels = [0, 2, 1, 7, 9, 6, 2, 3, 4, 5, 8, 7, 7]
        # Confusion matrix
        print('\nCreating confusion matrix...')
        conf_mat = confusion_matrix(self.org_labels,self.pred_labels)
        # Plot Confusion Matrix
        plt.figure()
        self.plotConfusionMatrix(conf_mat, classes=self.class_names, normalize=True, title='Normalized confusion matrix')
        plt.show()


    def createNewFile(self):
        if self.camera_flag == 0:
            fname, filter = QFileDialog.getSaveFileName(self, "Save File", "BangDiem_.xls", "Excel workbook (*.xls)")
            if fname:
                workbook = xlwt.Workbook(encoding="utf-8")  # tao file
                worksheet = workbook.add_sheet("Sheet 1")  # tao trang
                worksheet.write(0, 0, 'STT')
                worksheet.write(0, 1, "MSSV")
                worksheet.write(0, 2, "Diem Tong")
                worksheet.write(0, 3, "Diem Rubric")
                worksheet.col(0).width = 256 * 4
                worksheet.col(1).width = 256 * 15
                worksheet.col(2).width = 256 * 15
                worksheet.col(3).width = 256 * 15
                for rangeList in range(len(self.MssvList)):
                    worksheet.write(rangeList + 1, 0, rangeList + 1)
                    worksheet.write(rangeList + 1, 1, self.MssvList[rangeList])
                    worksheet.write(rangeList + 1, 2, self.ScoresList[rangeList])
                    worksheet.write(rangeList + 1, 3, self.RubricList[rangeList])
                workbook.save(fname)
            else:
                print('Error')


    def addFile(self):
        if self.camera_flag == 0:
            fname, filter = QFileDialog.getOpenFileName(self, "Open File", "BangDiem_.xls")
            if fname:
                workbook_read = xlrd.open_workbook(fname)  # doc file excel
                worksheet_read = workbook_read.sheet_by_index(0)
                total_rows = worksheet_read.nrows
                total_cols = worksheet_read.ncols
                workbook_write = copy(workbook_read)
                worksheet_write = workbook_write.get_sheet(0)
                for rangeList in range(len(self.MssvList)):
                    worksheet_write.write(rangeList + total_rows, 0, rangeList + total_rows - 1)
                    worksheet_write.write(rangeList + total_rows, 1, self.MssvList[rangeList])
                    worksheet_write.write(rangeList + total_rows, 2, self.ScoresList[rangeList])
                    worksheet_write.write(rangeList + total_rows, 3, self.RubricList[rangeList])
                workbook_write.save(fname)
            else:
                print('Error')


    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if (len(img.shape) == 3):
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.originalLabel.setPixmap(QPixmap.fromImage(outImage))
            self.originalLabel.setScaledContents(True)
        if window == 2:
            self.tableLabel.setPixmap(QPixmap.fromImage(outImage))
            self.tableLabel.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BangDiemGUI()
    window.setWindowTitle('Bang Diem')
    window.show()
    sys.exit(app.exec_())
