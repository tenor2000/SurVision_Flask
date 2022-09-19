import os
import cv2
import numpy as np

import pytesseract  # for reading the form ID #

# still having trouble finding relative route
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\gregj\PycharmProjects\SurVision_Flask\extract\Tesseract-OCR\tesseract.exe"

###### Image conversion Function ########
def getSurveyFiles(images, FileNum, pageCount, savePath):
    if pageCount >= len(images):
        print("Conversion Complete")
        return None

    num = "0000"
    num += str(FileNum)
    num = num[-4:]

    # Page One
    img = images[pageCount - 1]
    img.save(f"{savePath}/{num}A.png")
    # img = cv2.imread(f'{savePath}/{num}A.png')
    # cv2.imshow("Image", img)
    # cv2.waitKey(50)

    # Page Two
    img2 = images[pageCount]
    img2.save(f"{savePath}/{num}B.png")
    # img = cv2.imread(f'{savePath}/{num}B.png')
    # cv2.imshow("Image", img)
    # cv2.waitKey(50)

    return getSurveyFiles(images, FileNum + 1, pageCount + 2, savePath)


### File Naming Functions
def rectContour(contours, areaThresh):
    rectCon = []
    # filter out small rectangles by area
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > areaThresh:
            # print(area)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print("Corner Points",len(approx))
            if len(approx) == 4:
                rectCon.append(i)
    # creates a list of rectangular contours that are over the areaThresh
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)

    return rectCon


# find the corner points of rectangles
def getCornerPoints(rectangle):
    peri = cv2.arcLength(rectangle, True)
    approx = cv2.approxPolyDP(rectangle, 0.02 * peri, True)
    a, b, c, d = approx  # takes 4 points and assigns them to variable
    # reorder for the repositioning
    # if box leans left
    if (a[0][0] + a[0][1]) > (b[0][0] + b[0][1]):
        return b[0], a[0], c[0], d[0]
    # if box leans right
    else:
        return a[0], d[0], b[0], c[0]


def readIDfromIMG(img, width, height):
    # Preprocessing
    imgContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    # Finding all contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # Find rectangles
    rectCont = rectContour(contours, 40000)  # Its sorted from biggest area to smallest
    smallerCont = getCornerPoints(rectCont[1])

    pts1 = np.float32([smallerCont])  # find corner points (top left, top right, bot left, bot right)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgCrop = cv2.warpPerspective(img, matrix, (width, height))
    imgCrop2 = imgCrop[0:200, 650:]
    # cv2.imshow("form box", imgCrop2)
    # cv2.waitKey(50)
    formID = pytesseract.image_to_string(imgCrop2)

    # find and keep only digits and leave out the rest
    digitList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    listNum = [x for x in formID if x in digitList]
    resultNum = int(''.join(listNum))

    return resultNum


def readIDfromIMGnobox(img):
    imgCrop = img[1150:1275, 0:200]
    # cv2.imshow("form box", imgCrop)
    # cv2.waitKey(50)
    formID = pytesseract.image_to_string(imgCrop)
    # find and keep only digits and leave out the rest
    digitList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    listNum = [x for x in formID if x in digitList]
    resultNum = int(''.join(listNum))

    return resultNum

def rename_files(directory):
    for i in range(0, len(directory), 2):
        try:
            img = cv2.imread(directory[i])
            img2 = cv2.imread(directory[i + 1])
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # cv2.imshow("img", img)
            # cv2.waitKey(30)
            # id = readIDfromIMG(img, width, height)
            id = readIDfromIMGnobox(img)
            new_nameA = str(id) + "A.png"
            new_nameB = str(id) + "B.png"
            os.rename(directory[i], new_nameA)
            cv2.imwrite(f"{new_nameA}", img)
            os.rename(directory[i + 1], new_nameB)
            cv2.imwrite(f"{new_nameB}", img2)

        except:
            print("Error encountered")
            img = cv2.imread(directory[i])
            img2 = cv2.imread(directory[i + 1])
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # cv2.imshow("img", img)
            # cv2.waitKey(30)
            num = "0000" + str(i)
            new_nameA = f"unknown({num[-6:]})A.png"
            new_nameB = f"unknown({num[-6:]})B.png"
            os.rename(directory[i], new_nameA)
            cv2.imwrite(f"{new_nameA}", img)
            os.rename(directory[i + 1], new_nameB)
            cv2.imwrite(f"{new_nameB}", img2)
