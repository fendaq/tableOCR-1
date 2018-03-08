# coding=utf-8
import copy
import numpy as np
import os

import cv2

import preprocess


class TableHorLine:
    def __init__(self):
        self.line = preprocess.LineParam([0, 0, 1, 1])
        self.pts = [] # 横线与竖线的角点，即grid端点


def comparePicxelRed(pixel, thresh=150):
    newpix = [255,255,255]
    R = [0,0,255]
    if np.fabs((int)(pixel[2])-(int)((int)(pixel[0])+(int)(pixel[1]))/2) > 20\
            and ((int)(pixel[0])+(int)(pixel[1]))/2>100:
        return newpix
    else:
        return pixel

    # if np.fabs((int)(pixel[2]) - (int)((int)(pixel[0]) + (int)(pixel[1])) / 2) > 20:
    #     newpix = [pixel[0],pixel[1],(int)((pixel[0]+pixel[1])/2)]
    #     return newpix
    # else:
    #     return pixel
    # distance = np.sqrt((pixel[0]-R[0])**2+(pixel[1]-R[1])**2+(pixel[2]-R[2])**2)
    # if distance<thresh:
    #     return newpix
    # else:
    #     return pixel

def enhanceImg(image):
    # 去除红色通道
    for y,row in enumerate(image):
        for x,col in enumerate(row):
            image[y][x]=comparePicxelRed(image[y][x]) # col=image[y][x]

    # 去噪声
    image = cv2.medianBlur(image,3)
    return image

def get_ctpn(imgPath, txtPath, resultPath):
    for imgname in os.listdir(imgPath):
        # try:
        txt = 'res_' + os.path.splitext(imgname)[0] + '.txt'
        img = cv2.imread(os.path.join(imgPath, imgname))

        pts = []
        f = open(os.path.join(txtPath, txt))
        while (True):
            line = f.readline()  # str(min_x),str(min_y),str(max_x),str(max_y)
            line = line.strip('\r\n')
            if line == None or line == '':
                break
            coords = line.split(',', 3)
            x1 = int(coords[0]);
            y1 = int(coords[1]);
            x2 = int(coords[2]);
            y2 = int(coords[3])
            p1 = [x1, y1];
            p2 = [x2, y1];
            p3 = [x2, y2];
            p4 = [x1, y2]
            rect = [p1, p2, p3, p4]
            pts.append(rect)
        cropedImgs = preprocess.getTextImgCrop(img, pts)
        if cropedImgs!=None:
            for idx,image in enumerate(cropedImgs):
                cv2.imwrite(os.path.join(resultPath,os.path.splitext(imgname)[0]+str(idx)+'.jpg'), image)
        # except:
        #     print 'error: def get_ctpn(): !!!'

def mergeLines(lines,rhoThresh=7,thetaThresh=0.09):
    dictMerg = {} # dict = {int lineIndex, bool isMerged}
    Lines = []
    mergedLines = []
    for index,line in enumerate(lines[0]):
        myline = preprocess.LineParam(tuple(line))
        Lines.append(myline)
        dictMerg[index] = False

    #debug
    debugline = []
    for line in Lines:
        if np.abs(line.line[1]-np.pi/2) < thetaThresh:
            debugline.append(line)
    debugline.sort(key=lambda x:x.line[0])

    for i in range(len(Lines)):
        if dictMerg[i]==True:
            continue
        for j in range(i+1,len(Lines)):
            if dictMerg[j]==True:
                continue
            if np.abs(Lines[i].line[0]-Lines[j].line[0])<rhoThresh and np.abs(Lines[i].line[1]-Lines[j].line[1])<thetaThresh:
                # 合并直线
                pts = [Lines[i].p1, Lines[i].p2, Lines[j].p1, Lines[j].p2]
                if Lines[i].line[1]>np.pi/4 and Lines[i].line[1]<np.pi/4*3:
                    pts.sort()
                else:
                    pts.sort(key=lambda x:x[1])

                Lines[i] = preprocess.LineParam([pts[0][0], pts[0][1], pts[3][0], pts[3][1]])
                dictMerg[j] = True
        mergedLines.append(Lines[i])

    return mergedLines

def selectLines(horlines, binaryImg):
    horlines.sort(key=lambda x:x.line[0])
    imgW = binaryImg.shape[1]
    imgH = binaryImg.shape[0]
    HorLines=[]
    # 记录哪些线要被删除
    hordict = {}
    gray = cv2.cvtColor(binaryImg,cv2.COLOR_GRAY2BGR)
    _,binaryImg = cv2.threshold(binaryImg,0.0,255.0,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    for index,line in enumerate(horlines):
        # cv2.line(gray, (line.p1[0], line.p1[1]), (line.p2[0], line.p2[1]), (255, 0, 0), 1)
        hordict[index] = True
    # 横线
    threshHor = 0.25
    ratiosHor = []
    for index, line in enumerate(horlines):
        Xs = []  # 两侧有文字的坐标集
        delt = 1  # 向两侧寻找的范围
        Xstart = min(line.p1[0], line.p2[0])
        Xend = max(line.p1[0], line.p2[0])
        for x in range(Xstart, Xend):
            y = line.getY(x)
            if y - delt < 0 or y + delt > imgH:
                continue
            flag1 = False
            flag2 = False
            for i in range(2, 2+delt):
                if binaryImg[y+i][x] > 80:
                    flag1 = True
                if binaryImg[y-i][x] > 80:
                    flag2 = True
            if flag1 and flag2:
                Xs.append(x)
        ratio = (float)(len(Xs)) / (float)(np.abs(line.p1[0] - line.p2[0]))
        ratiosHor.append(ratio)
        if ratio > threshHor:
            hordict[index] = False

    for index,line in enumerate(horlines):
        if hordict[index]:
            HorLines.append(line)
            cv2.line(gray, (line.p1[0], line.p1[1]), (line.p2[0], line.p2[1]), (255, 0, 0), 1)
        else:
            cv2.line(gray, (line.p1[0], line.p1[1]), (line.p2[0], line.p2[1]), (0, 0, 255), 1)

    return HorLines

def selectLineByInterval(horlines):
    HorLines = []
    intervals = []
    mergeThresh = 5
    centers = [] # 存interval聚类中心 center=[interval, num]
    for i in range(0,len(horlines)-1):
        intervals.append(horlines[i+1].line[0]-horlines[i].line[0])
    # 中心聚类
    center = [intervals[0], 1]
    centers.append(center)
    for i,interval in enumerate(intervals):
        flag = False
        for j,c in enumerate(centers):
            if np.abs(interval - c[0])<mergeThresh:
                num = c[1]
                value = c[0]
                centers[j][0] = (value*num + interval)/(num+1)
                centers[j][1] += 1
                flag = True
                break
        if flag==False:
            center = [intervals[i], 1]
            centers.append(center)
    centers.sort(key=lambda x:x[1], reverse=True)
    intervalMost = centers[0][0]
    # 间隔小于interval的线被删除
    dict = {}
    for i in range(len(horlines)):
        dict[i] = True
    for i,interval in enumerate(intervals):
        if interval<0.75*intervalMost:
            line1 = horlines[i]
            line2 = horlines[i+1]
            if i+1 < len(intervals):
                if intervals[i+1]<0.75*intervalMost:
                    dict[i+1]=False
    for i in range(len(horlines)):
        if dict[i]:
            HorLines.append(horlines[i])

    return HorLines

def findLine(image):
    # 预处理图片
    imgW = image.shape[1]
    imgH = image.shape[0]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = preprocess.convert(gray)
    binary = copy.copy(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # 提取线段
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=250,minLineLength=0.1*min(imgH,imgW),maxLineGap=0.02*min(imgH,imgW))
    # print 'line num ===',len(lines[0])
    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    # 合并重叠的线
    mergLines = mergeLines(lines)
    # print 'merged line num ==',len(mergLines)
    # 画图
    # for idx,line in enumerate(mergLines):
    #     if line.p1[0]<10:
    #         print 'debug'
    #     cv2.line(edges,(line.p1[0],line.p1[1]),(line.p2[0],line.p2[1]),(255,0,0),2)

    # 分垂线横线
    veclines = []
    horlines = []
    mergLines.sort(key=lambda x:x.line[0])
    # angThresh = np.pi/9
    angThresh = 0.028
    for index,line in enumerate(mergLines):
        # if index==5:
        #     # print 'debug'
        if line.line[1] < angThresh or np.fabs(line.line[1] - np.pi) < angThresh:  # 垂直方向筛选
            # 删除页面边界线
            if np.abs(line.line[0])<5.0:
                continue
            veclines.append(line)
        elif np.fabs(line.line[1]-np.pi/2)<angThresh:# 水平方向筛选
            # 删除页面边界线
            if np.abs(line.line[0])<5.0:
                continue
            horlines.append(line)

    # 按照长度剔除杂线
    ratio = 0.25
    horlines.sort(key=lambda x:x.length,reverse=True)
    for idx,line in enumerate(horlines):
        if horlines[idx].length < ratio*horlines[0].length:
            horlines = horlines[0:idx]
            break
    # 垂线只保留边界线
    veclines.sort(key=lambda x: x.p1[0], reverse=True)
    VecLines = [veclines[0],veclines[len(veclines)-1]]

    # 剔除横穿文字的线段
    HorLines = selectLines(horlines,binary)
    veclines = VecLines
    horlines = HorLines

    # 按照横线间隔规律 剔除横线
    horlines = selectLineByInterval(horlines)

    # 画图
    for line in veclines:
        cv2.line(edges, (line.p1[0], line.p1[1]), (line.p2[0], line.p2[1]), (255, 0, 0), 2)
    for line in horlines:
        cv2.line(edges, (line.p1[0], line.p1[1]), (line.p2[0], line.p2[1]), (255, 0, 0), 2)

    # 排序，选出表格区域
    # veclines.sort(key=lambda x:x.p1[0])
    # horlines.sort(key=lambda x:x.line[0])
    return image, edges, veclines, horlines

def gridRestrain(image, edges, grid):
    # p1--p2
    # p3--p4
    for pt in grid:
        cv2.circle(edges,tuple(pt),4,(0,0,255),2)
    x1 = (int)((grid[0][0]+grid[2][0])/2)
    x2 = (int)((grid[1][0]+grid[3][0])/2)
    y1 = (int)((grid[0][1]+grid[1][1])/2)
    y2 = (int)((grid[2][1]+grid[3][1])/2)
    if np.abs(grid[0][0]-grid[2][0])>3 or np.abs(grid[1][0]-grid[3][0])>3 \
            or np.abs(grid[0][1]-grid[1][1])>3 or np.abs(grid[2][1]-grid[3][1])>3:
        src = np.float32(grid)
        des = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
        M = cv2.getPerspectiveTransform(src,des)
        imageNew = cv2.warpPerspective(image,M,(image.shape[1],image.shape[0]))
        if y2>y1 and x2>x1:
            gridImg = np.zeros((y2-y1,x2-x1,3),np.uint8)
            gridImg = imageNew[y1:y2, x1:x2]
        else:
            gridImg = None
    else:
        if y2 > y1 and x2 > x1:
            gridImg = image[y1:y2, x1:x2]
        else:
            gridImg = None
    return edges, gridImg


def getGrid(image, edges, veclines, horlines):
    tablelines = []
    # 横线从上到下，垂线从左到右，求交点
    horlines.sort(key=lambda x:x.line[0])
    veclines.sort(key=lambda x:x.p1[0]) # vecline rho 可能为负数
    for horline in horlines:
        tableline = TableHorLine()
        for vecline in veclines:
            point = preprocess.lineCrossPoint(horline, vecline)
            tableline.pts.append(point)
        tablelines.append(tableline)

    # 切片，画grid角点，矫正grid
    Grids = []
    for i in range(len(tablelines)-1):
        line1 = tablelines[i]
        line2 = tablelines[i+1]
        numPt = min(len(line1.pts),len(line2.pts))
        for j in range(numPt-1):
            p1 = line1.pts[j]
            p2 = line1.pts[j+1]
            p3 = line2.pts[j]
            p4 = line2.pts[j+1]
            grid = [p1,p2,p3,p4]
            edges, gridImg = gridRestrain(image, edges, grid)
            if not gridImg is None:
                Grids.append(gridImg)

    return edges,Grids

def divideGrid(linegridImg):
    grid = []
    # 变为白字黑底
    gray = cv2.cvtColor(linegridImg,cv2.COLOR_BGR2GRAY)
    gray = preprocess.convert(gray)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bw = cv2.dilate(gray,kernel,iterations=1)
    _, bw = cv2.threshold(bw, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    w = linegridImg.shape[1]
    h = linegridImg.shape[0]
    sums = []
    xcoord = []
    threshSumHigh = 255*h*0.95
    threshSumLow = 255*h*max(0.2,(float)(6)/(float)(h))
    count = 0
    for rol in range(w):
        pixsum = sum(bw[:, rol])
        # print 'rol:', rol, ' sum:', pixsum
        sums.append(pixsum)

    for rol in range(2, linegridImg.shape[1] - 3):
        if (sums[rol - 2] > threshSumHigh or sums[rol + 2] >= threshSumHigh) and sums[rol] <= threshSumLow:
            xcoord.append(rol)
        else:
            continue

    # print 'x border = ', xcoord
    # for x in xcoord:
    #     cv2.line(linegridImg, (x, 0), (x, linegridImg.shape[0]), (0, 0, 255), 2)
    # cv2.imshow('result', image)
    # cv2.waitKey(10)
    #
    # cv2.imshow('bw',bw)
    #
    # key = cv2.waitKey(0)
    # if key>0:
    #     return xcoord
    return xcoord

#wz
def getRevisedTextLine(src_line_img):
    gray = copy.deepcopy(src_line_img)
    if len(src_line_img.shape) > 1:
        gray = cv2.cvtColor(src_line_img, cv2.COLOR_BGR2GRAY)


    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess_line_img(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)


    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(src_line_img, [box], 0, (0, 255, 0), 2)

    # cv2.namedWindow("countour", cv2.WINDOW_NORMAL)
    # cv2.imshow("countour", src_line_img)
    # cv2.waitKey(0)

    return region

def preprocess_line_img(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # sobel = cv2.medianBlur(sobel, 5)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    # # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)
    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(binary, element1, iterations = 1)
    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations = 1)

    return dilation2

def findTextRegion(img):
    region = []
    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if(area < 10):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)


        # box是四个点的坐标
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if(height > width * 1.2):
            continue

        region.append(box)

    return region


def get_text_projection(srcimg, is_horizontal = True):
    '''
    :param srcimg:
    :param is_horizontal: if False, take vertical projection
    :return: list of projection number
    '''
    height = srcimg.shape[0]
    width = srcimg.shape[1]
    gray = copy.deepcopy(srcimg)
    if is_horizontal:
        counter = [0] * height
        if len(gray.shape) > 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        retval, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imshow('binary', binary)
        # cv2.waitKey(0)
        for i in range(height):
            number = 0
            for j in range(width):
                if binary[i][j] == 0:
                    number += 1
            counter[i] = number
        return counter
    else:
        counter = [0] * width
        if len(gray.shape) > 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        retval, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('binary', binary)
        # cv2.waitKey(0)
        for i in range(width):
            number = 0
            for j in range(height):
                if binary[j][i] == 0:
                    number += 1
            counter[i] = number
        return counter


def get_peek_range(proj_counter, min_thresh = 2, min_range = 10):
    '''
    :param proj_counter: the projection list(horizontal or vertical)
    :param min_thresh: min proj range to be regarded as a char stroke.
    :param min_range: a char's range.
    :return: the list of [begin ,end] position pair of chars.
    '''
    begin = 0
    end = 0
    char_range_list = []
    for i in range(len(proj_counter)):
        if proj_counter[i] > min_thresh and begin == 0:
            begin = i
        elif proj_counter[i] > min_thresh and begin != 0:
            continue
        elif proj_counter[i] < min_thresh and begin != 0:
            end = i
            if end - begin >= min_range:
                char_range_list.append([begin, end])
                begin = 0
                end = 0
        elif proj_counter[i] < min_thresh or begin == 0:
            continue

    return char_range_list

def get_proj_img(proj_counter, is_horizontal = True):
    '''
    :param proj_counter:  the projection list(horizontal or vertical)
    :param is_horizontal:
    :return: type: numpy array, projection image.
    '''
    if is_horizontal:
        height = len(proj_counter)
        width = max(proj_counter)
        project = np.zeros([height, width], np.int)
        for r in range(height):
            for c in range(proj_counter[r]):
                project[r][c] = 255
        return project
    else:
        width = len(proj_counter)
        height = max(proj_counter)
        project = np.zeros([height, width], np.int)
        for c in range(width):
            for r_ in range(proj_counter[c]):
                r = height - r_ - 1
                project[r][c] = 255
        return project


def thresholdtest(img):
    image = copy.deepcopy(img)
    if len(image.shape) > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    return th

if __name__=='__main__':
    imgdir = 'data/ctpnGrid/1111'
    output_path = 'data/revised_text_line'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for image_index, img_name in enumerate(os.listdir(imgdir)):
        img_path = os.path.join(imgdir, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_CUBIC)

        prj_counter_horizon = get_text_projection(image, is_horizontal = True)
        pos_horizon = get_peek_range(prj_counter_horizon)

        # projimg = get_proj_img(prj_counter_horizon)

        for pt in pos_horizon:
            begin = pt[0]
            end = pt[1]
            sub_line = image[begin: end, :, :]
            # cv2.imshow('sub_line', sub_line)
            # cv2.waitKey(0)

            prj_counter_vertical = get_text_projection(sub_line, is_horizontal = False)
            pos_vertical = get_peek_range(prj_counter_vertical)

            # projimg_v = get_proj_img(prj_counter_vertical)
            # cv2.imshow('projimg_v', projimg_v)
            # cv2.waitKey(0)

            print(prj_counter_vertical)
            print(pos_vertical)

            if len(pos_vertical) == 0:
                continue

            text_region = sub_line[:, pos_vertical[0][0]: pos_vertical[-1][1], :]
            cv2.imshow('text', text_region)
            cv2.waitKey(0)

        # th = thresholdtest(image)
        # cv2.imshow('th', th)
        # cv2.waitKey(0)
        # text_boxes = getRevisedTextLine(image)


        # for index, box in enumerate(text_boxes):
        #     minx, miny = box.min(0)
        #     maxx, maxy = box.max(0)
        #
        #     minx = minx - 2
        #     miny = miny - 2
        #     maxy = maxy + 2
        #     maxx = maxx + 2
        #
        #     if minx < 0: minx = 0
        #     if miny < 0: miny = 0
        #     if maxx >= image.shape[1]: max = image.shape[1] - 1
        #     if maxy >= image.shape[0]: max = image.shape[0] - 1
        #     subimg = image[miny: maxy, minx: maxx, :]
        #     cv2.imwrite(os.path.join(output_path, str(image_index) + '_' + str(index) + '.jpg'), subimg)
        #     cv2.imshow('subimg', subimg)
        #     cv2.waitKey(0)