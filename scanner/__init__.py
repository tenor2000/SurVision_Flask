import os
import cv2
import numpy as np
import pandas as pd
from SurVision import app
from tkinter import *
from PIL import ImageTk, Image
import time

# Troubleshooting
print_dataline = True
show_error_img = False
double_checker = False
print_answer_blocks = False

# set up for Tkinter (will be deleted)
root = Tk()
root.title("Processing images")

rootLabel = Label(root, text="Human, do not delete me or I will malfunction.")
rootLabel.pack()

# some residual from old setup.
altanswer_questions1 = ["C", "G", "H"]
altanswer_questions2 = ["C", "H", "I"]
altanswer_choices = [1, 2, 9, 0]

### Functions ####
# Convert to grayscale.
def rectContour(contours, areaThreshLow, areaThreshHigh):
    rect_con = []
    # filter out unwanted rectangles by area
    for i in contours:
        area = cv2.contourArea(i)
        if area > areaThreshLow and area < areaThreshHigh:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print("Corner Points",len(approx))
            if len(approx) == 4:
                rect_con.append(i)
    # creates a list of rectangular contours that are over the areaThresh
    rect_con = sorted(rect_con, key=cv2.contourArea, reverse=True)

    return rect_con


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

def processIMG(img, cropdim):
    # Preprocessing
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)
    # Finding all contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Find rectangles
    # Its sorted from biggest area to smallest with threshold area of at least arg  2 (40000)
    rectCont = rectContour(contours, 40000, float('inf'))
    biggestCont = getCornerPoints(rectCont[0])

    # FIND new dimensions from cropping down (work in progress)
    width, height = cropdim

    pts1 = np.float32([biggestCont])  # find corner points (top left, top right, bot left, bot right)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # reset the corner points to new size
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_crop = cv2.warpPerspective(img, matrix, (width, height))
    # Remove black border
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    boxWidth = 5
    x, y, w, h = cv2.boundingRect(cnt)
    img_crop = img_crop[y + boxWidth:y + h - boxWidth, x + boxWidth:x + w - boxWidth]

    return img_crop

def create_mask(img):
    # Adjust contrast settings and gate in pixels (hue, saturation, value)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, 220])
    # final processed image for scanning results
    mask = cv2.inRange(imgHSV, lower, upper)
    return mask

def match_image(img1, img2):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(img1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches = list(matches)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = img2.shape
    img1Reg = cv2.warpPerspective(img1, h, (width, height))

    return img1Reg, h

# AnswerKey just needs a few inputs and does everything automagically
class AnswerKey:
    def __init__(self, img, cropdim, ansnums, alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.img = processIMG(img, cropdim)
        self.question_num = len(ansnums)
        self.ansnums = ansnums
        self.alpha = alpha
        self.checkboxes = self.find_checkbox()
        # unsorted boxes sorted to left and right then recombined
        self.boxlist = self.sort_boxes(self.checkboxes, cropdim[0])
        # EXPORT this. Assigned letters to questions and boxes
        self.question_assignments = self.assign_questions()
        if sum(self.ansnums) != len(self.boxlist):
            raise Exception("The number of answers and the number of checkboxes do not match")

    def assign_questions(self):
        dictpage = dict()
        used_ranges = []
        # for 'Mark All That Apply' questions
        subcounter = 1

        for x in range(self.question_num):
            # Mark all that apply, each answer is a separate question.
            if self.ansnums[x] == 1:
                dictpage[self.alpha[0] + str(subcounter)] = self.boxlist[
                                                            sum(used_ranges):self.ansnums[x] + sum(used_ranges)]
                used_ranges.append(self.ansnums[x])
                subcounter += 1
            else:
                # Complete the Mark all that apply Question and go on to the next letter.
                if len(used_ranges) > 0 and used_ranges[-1] == 1:
                    ## Combine All Mark that Apply checkboxes for reference use, will mess up scoring though.
                    # temp = [dictpage[x] for x in dictpage.keys() if x[0] == self.alpha[0]]
                    ## reduce the list of lists
                    # flattemp = [item for sublist in temp for item in sublist]
                    # dictpage[self.alpha[0]] = flattemp
                    subcounter = 1  # just resets counter for Mark all that apply
                    self.alpha = self.alpha[1:]
                temp = self.boxlist[sum(used_ranges):self.ansnums[x] + sum(used_ranges)]
                # compare first boxes, if they are far apart on x axis, this signifies two columns.
                a = temp[0][0][0]  # 1st checkbox, 1st coordinate, x value
                b = temp[1][0][0]  # 2nd checkbox, 1st coordinate, x value
                if abs(a - b) > 50:
                    temp = self.sort_boxes(temp, a + b)
                # assign new dictionary entry for question
                dictpage[self.alpha[0]] = temp
                used_ranges.append(self.ansnums[x])
                self.alpha = self.alpha[1:]

        return dictpage

    def find_checkbox(self):
        imgGray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 21)

        # Finding all contours
        contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rectCont = rectContour(contours, 200, 500)

        # find corners
        boxlist = []
        for box in rectCont:
            boxlist.append(getCornerPoints(box))
        return boxlist

    def sort_boxes(self, boxlist, width):
        left = []
        right = []
        for box in boxlist:
            x1, y1 = box[0]
            if x1 < width / 2:
                left.append(box)
            else:
                right.append(box)
        # sort by y axis first. then during individual questions sort by x (for two columns)
        left = sorted(left, key=lambda x: x[0][1])
        right = sorted(right, key=lambda x: x[0][1])

        return left + right

    def show_checkboxes(self, questionID=None):
        imgContour = self.img.copy()

        if questionID == None:
            for box in self.checkboxes:
                cv2.rectangle(imgContour, box[0], box[3], (0, 255, 0), 2)
        else:
            try:
                for box in self.question_assignments[questionID]:
                    cv2.rectangle(imgContour, box[0], box[3], (0, 255, 0), 2)
            except:
                print(f"{questionID} does not appear to exist, try again.")
        # cv2.namedWindow('Question', cv2.WINDOW_NORMAL)
        cv2.imshow("Question", imgContour)
        cv2.waitKey(0)

# for marking Surveys using answerKey
class Questions:
    def __init__(self, img, keyimg, cropdim, boxlist):
        self.img0 = processIMG(img, cropdim)
        self.img, h = match_image(self.img0, keyimg)
        self.mask = create_mask(self.img)
        self.boxlist = boxlist  # dict

    # Original Answer Getter based of of threshold and pixel count
    def get_answer_old(self, questionID, altanswers=[]):
        # questionID is a str
        answer = []
        for i, box in enumerate(self.boxlist[questionID]):
            # crop image of an answer square
            x1, y1 = box[0]
            x2, y2 = box[1]
            x3, y3 = box[2]
            x4, y4 = box[3]

            top_left_x = min([x1, x2, x3, x4])
            top_left_y = min([y1, y2, y3, y4])
            bot_right_x = max([x1, x2, x3, x4])
            bot_right_y = max([y1, y2, y3, y4])
            # exclude the square itself
            bordersize = 2

            square = self.mask[top_left_y + bordersize:bot_right_y - bordersize + 1,
                     top_left_x + bordersize:bot_right_x - bordersize + 1]  # square
            if print_answer_blocks == True:
                print(f"{questionID}{i+1}: " + str(np.sum(square == 255)))
                cv2.imshow(f"{questionID}{i+1}", square)
                #cv2.waitKey(0)
            if np.sum(square == 255) >= upperThresh:
                if len(altanswers) > 0:
                    try:
                        answer.append(altanswers[i])
                    except:
                        # print("altanswer")
                        return self.doublecheck(questionID)
                else:
                    answer.append(i + 1)
            elif np.sum(square == 255) >= lowerThresh:
                # print("threshold")
                return self.doublecheck(questionID)

        if len(answer) > 1:
            # print("more than 1")
            return self.doublecheck(questionID)
        elif len(answer) < 1:
            answer.append(None)
        return answer[0]

    #New Get answer based of average of lower vs higher.
    def get_answer(self, questionID, altanswers=[], maxgap=25, mingap=15):
        # questionID is a str
        pixel_counts = []
        for i, box in enumerate(self.boxlist[questionID]):
            # crop image of an answer square
            x1, y1 = box[0]
            x2, y2 = box[1]
            x3, y3 = box[2]
            x4, y4 = box[3]

            top_left_x = min([x1, x2, x3, x4])
            top_left_y = min([y1, y2, y3, y4])
            bot_right_x = max([x1, x2, x3, x4])
            bot_right_y = max([y1, y2, y3, y4])
            # inclusion of square and surrounding area
            bordersize = -3

            square = self.mask[top_left_y + bordersize:bot_right_y - bordersize + 1,
                     top_left_x + bordersize:bot_right_x - bordersize + 1]  # square
            if print_answer_blocks == True:
                print(f"{questionID}{i+1}: " + str(np.sum(square == 255)))
                cv2.imshow(f"{questionID}{i+1}", square)
            pixel_counts.append(np.sum(square == 255))

        counts = [x for x in pixel_counts]
        counts.sort()
        groups = [[counts[0]]]
        for x in counts[1:]:
            if x - min(groups[-1]) > maxgap:
                groups.append([x])
            elif abs(x - groups[-1][-1]) <= maxgap:
                groups[-1].append(x)
            else:
                groups.append([x])
        # print(groups)
        if len([x for x in counts if x > 250]) > 1:
            return self.doublecheck(questionID)
        elif len(groups[-1]) == 1:
            if groups[-1][0] - max(groups[-2]) < mingap:
                return self.doublecheck(questionID)
            answer_index = pixel_counts.index(groups[-1][0])
            if len(altanswers) > 0:
                try:
                    return altanswers[answer_index]
                except:
                    print("Actual answer can't match an alternate answer, please revise alternate answers")
                    return "ALT" + str(answer_index+1)
            return answer_index + 1
        elif len(groups) > 1 and len(groups[-1]) > 1:
            return self.doublecheck(questionID)
        elif len(groups) == 1 and max(groups[0])-min(groups[0]) > mingap:
            return self.doublecheck(questionID)
        else:
            return None

    def show_checkboxes(self, questionID=None):
        img_contour = self.img.copy()
        if questionID == None:
            temp = [self.boxlist[x] for x in self.boxlist.keys()]
            temp = [item for sublist in temp for item in sublist]
            for box in temp:
                cv2.rectangle(img_contour, box[0], box[3], (0, 255, 0), 2)
        else:
            try:
                for box in self.boxlist[questionID]:
                    cv2.rectangle(img_contour, box[0], box[3], (0, 255, 0), 2)
            except:
                print(f"{questionID} does not appear to exist, try again.")

        cv2.imshow("Question", img_contour)
        cv2.waitKey(0)

    def crop_question(self, image, questionID):
        left = self.boxlist[questionID][0][0][0] - 50
        right = self.boxlist[questionID][0][0][0] + 300
        upper = self.boxlist[questionID][0][0][1] - 50
        lower = self.boxlist[questionID][0][0][1] + 175

        # Establish smaller pic window for smaller questions
        if len(self.boxlist[questionID]) < 3:
            lower -= 75

        cropped_image = image[upper:lower, left:right]

        return cropped_image

    def doublecheck(self, questionID):
        if double_checker == False:
            if str(filename) not in checks:
                checks.append(f"{filename}")
            return (f'check Question {questionID}')

        # Draw boxes for the question
        img_copy = self.img.copy()
        for box in self.boxlist[questionID]:
            cv2.rectangle(img_copy, box[0], box[3], (0, 0, 255), 1)

        img_question = self.crop_question(img_copy, questionID)

        # Reorder colors to convert to tkinter/PIL image
        b, g, r = cv2.split(img_question)
        img1 = cv2.merge((r, g, b))
        img2 = Image.fromarray(img1)

        imgtk = ImageTk.PhotoImage(image=img2)

        pop = Toplevel(root)
        pop.title(f"Question {questionID}")

        def onClick(event=None):
            pop.quit()

        pop.bind('<Return>', onClick)

        quest_img = Label(pop, image=imgtk)
        quest_img.pack()
        txt = Label(pop, text=f"Human, I require your immediate assistance with Question {questionID}.")
        txt.pack()
        txt2 = Label(pop, text="Enter a number or leave it blank.")
        txt2.pack()
        entry = Entry(pop, takefocus=True)
        entry.pack()
        entry.focus()
        button = Button(pop, text="OK", command=onClick)
        button.pack()
        pop.focus_force()
        pop.grab_set()  # for disable main window
        pop.attributes('-topmost', True)  # for focus on toplevel
        pop.mainloop()
        human_check = entry.get()
        pop.destroy()

        try:
            return int(human_check)
        except:
            pass

def get_results(Questions, dataline):
    results = []

    for question in Questions.boxlist:
        # Alternate Answer Questions
        if question in altanswer_questions1 and dataline[1] == 'Version 1':
            results.append(Questions.get_answer(question, altanswers=altanswer_choices))
        elif question in altanswer_questions2 and dataline[1] == "Version 2":
            results.append(Questions.get_answer(question, altanswers=altanswer_choices))
        else:
            results.append(Questions.get_answer(question))
    return results

def create_key(key_file_name, version_num, cropdimA, cropdimB):
    df = pd.read_csv(os.path.join(app.config['SURVEY_KEYS_FOLDER'], key_file_name))
    df = df[df["Version"] == version_num]

    imgA_path = os.path.join(app.config['SURVEY_KEYS_FOLDER'], df.iloc[0]['ImageA'])
    imgB_path = os.path.join(app.config['SURVEY_KEYS_FOLDER'], df.iloc[0]['ImageB'])
    key_imgA = cv2.imread(imgA_path, cv2.IMREAD_COLOR)
    key_imgB = cv2.imread(imgB_path, cv2.IMREAD_COLOR)

    # the cell in the key file must be string of numbers separated by commas
    answer_nums_listA = [int(x) for x in list(df.iloc[0]['QA'].split(', '))]
    answer_nums_listB = [int(x) for x in list(df.iloc[0]['QB'].split(', '))]

    answer_key_A = AnswerKey(key_imgA, cropdimA, answer_nums_listA)
    answer_key_B = AnswerKey(key_imgB, cropdimB, answer_nums_listB, alpha=answer_key_A.alpha)

    return answer_key_A, answer_key_B

def execute_survey_scanning(key_file_name):


    # Stuff that should be set by user
    cropdimA = (750, 935)
    cropdimB = (750, 1106)

    path_dir = app.config['UPLOAD_FOLDER']
    version = True

    papersize = "1"
    paperorient = "portrait"

    blank_thresh = 5

    layouts = {"0": (1275, 1650), "1": (825, 1275), "2": (1275, 2100)}
    origdim = layouts[papersize]
    if paperorient == "landscape":
        origdim = (origdim[1], origdim[0])

    altanswer_questions1 = ["C", "G", "H"]
    altanswer_questions2 = ["C", "H", "I"]
    altanswer_choices = [1, 2, 9, 0]

    key1A, key1B = create_key(key_file_name, 1, cropdimA, cropdimB)
    key2A, key2B = create_key(key_file_name, 2, cropdimA, cropdimB)

    # create dataframe to store information
    start_columns = ["FormID", "Version"]
    df1 = pd.DataFrame(
        columns=start_columns + list(key1A.question_assignments.keys()) + list(key1B.question_assignments.keys()))
    df2 = pd.DataFrame(
        columns=start_columns + list(key2A.question_assignments.keys()) + list(key2B.question_assignments.keys()))
    total_answers = df1.shape[1]
    total_answers2 = df2.shape[1]
    # Pixel Thresholds (old get_answers)
    upperThresh = 100
    lowerThresh = 80
    # Max pixel count gap (new get_answers)
    maxgap = 25
    mingap = 15
    fillinlim = 250
    # Lists of surveys that trigger an error, a double check or blank survey
    errors = []
    checks = []
    blanks = []
    # The dataline before its added to the dataframe
    dataline = []

    for filename in os.listdir(path_dir):
        try:
            if "A" in filename:
                # Process page 1
                dataline = []
                imgA = cv2.imread(f"{path_dir}/{filename}")
                #cv2.imshow("View", imgA)
                #cv2.waitKey(30)

                # Find ID number/Version Num found by file name

                formIDnum = filename[:-5]

                dataline.append(formIDnum)
                try:
                    if version:
                        if formIDnum % 2 == 0:
                            dataline.append("Version 2")
                        else:
                            dataline.append("Version 1")
                except:
                    if version:
                        if int(filename[:-5]) % 2 == 0:
                            dataline.append("Version 2")
                        else:
                            dataline.append("Version 1")

                # get results
                if dataline[1] == "Version 1":
                    PageOne = Questions(imgA, key1A.img, cropdimA, key1A.question_assignments)
                elif dataline[1] == "Version 2":
                    PageOne = Questions(imgA, key2A.img, cropdimA, key2A.question_assignments)
                dataline = dataline + get_results(PageOne, dataline)
            elif "B" in filename:
                # Process page 2
                imgB = cv2.imread(f"{path_dir}/{filename}")

                # get results
                if dataline[1] == "Version 1":
                    PageTwo = Questions(imgB, key1B.img, cropdimB, key1B.question_assignments)
                elif dataline[1] == "Version 2":
                    PageTwo = Questions(imgB, key2B.img, cropdimB, key2B.question_assignments)
                dataline = dataline + get_results(PageTwo, dataline)

                if print_dataline:
                    print(dataline)
                if dataline.count(None) > blank_thresh:
                    blanks.append(filename)
                if len(dataline) == total_answers and dataline[1] == "Version 1":
                    df1.loc[len(df1)] = dataline
                elif len(dataline) == total_answers2 and dataline[1] == "Version 2":
                    df2.loc[len(df2)] = dataline
                else:
                    print("Number of answers does not match expectation. Survey not recorded.")
                dataline = []
            else:
                print(f"Is {filename} a survey? If so, please check that name ends in 'A' or 'B' before the extension.")
                dataline = []  # Cleared for next question

        except IndexError:
            
            print(f"{filename} survey not tabulated most likely due to a contour processing ERROR.")
            errors.append(filename)
            if show_error_img:
                img = cv2.imread(f"{path_dir}/{filename}")
                img = cv2.resize(img, (700, 910))
                cv2.imshow(f"{filename} failed most likely due to contour ERROR", img)
                cv2.waitKey(0)
        except:
            print(f"{filename} survey not tabulated due to an ERROR.")
            errors.append(filename)
            if show_error_img:
                img = cv2.imread(f"{path_dir}/{filename}")
                img = cv2.resize(img, (700, 910))
                cv2.imshow(f"{filename} unknown ERROR", img)
                cv2.waitKey(0)

    df1.to_csv(f"{app.config['DOWNLOAD_FOLDER']}/resultsV1.csv", index=False)
    df2.to_csv(f"{app.config['DOWNLOAD_FOLDER']}/resultsV2.csv", index=False)
    print("\nMission Complete!")

    # Print checklist of surveys to be reviewed by humans
    if len(errors) > 0:
        print("\nAn Error has occurred while processing these files (Affected surveys were not tabulated):")
        for file in errors:
            print(file)
        print("Total:", len(errors))
    if len(checks) > 0:
        print("\nThe following files needed to be double checked for some answers")
        for file in checks:
            print(file)
        print("Total:", len(checks))
    if len(blanks) > 0:
        print("\nThe following files seem to be blank or many answers are not visible to me:")
        for file in blanks:
            print(file)
        print("Total:", len(blanks))
    end = time.time()

    #print(f"This run took {end - start} seconds to complete!")
