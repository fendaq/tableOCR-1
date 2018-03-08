# -*- coding:utf-8 -*-

import preprocess
import tableprocess
import cv2
import os
import codecs
import numpy as np
import copy
import _init_paths
from textdetection_hori.ctpn import ctpnport
from tesseract import tesseractport

#add by wz
from caffe_ocr import init_caffe_ocr_module as caffe_module
import xlsxwriter


this_dir = os.path.dirname(__file__)

#ctpn网络加载
ctpn_sess, ctpn_net = ctpnport.ctpnSource() #水平

#tesseract网络加载
tesseractport.tess_init_english()


def lineDetect(imgpath, rotate = False):
    # 直线检测 与 横条Grid 提取模块; 返回彩色横条grid和竖向分割坐标，保存中间结果
    image = cv2.imread(imgpath)
    if rotate:
        image = np.rot90(image)
        image = np.rot90(image)
        image = np.rot90(image)

    w = image.shape[1]
    h = image.shape[0]
    maxlen = 1920.0
    if w > h:
        image = cv2.resize(image, ((int)(maxlen), (int)(maxlen / w * h)))
    else:
        image = cv2.resize(image, ((int)(maxlen / h * w), (int)(maxlen)))

    # 表格处理
    image, edges, veclines, horlines = tableprocess.findLine(image)
    if len(veclines) < 2 or len(horlines) < 2:
        print '未检测到表格线，退出！'
        exit(0)
    edges, gridImgs = tableprocess.getGrid(image, edges, veclines, horlines)
    xcoords = []
    for index, grid in enumerate(gridImgs):
        if grid is None:
            print 'grid None'
            continue
        # 竖向分割坐标
        xcoord = tableprocess.divideGrid(grid)
        xcoords.append(xcoord)

    return gridImgs, xcoords


def singleGrid(gridImgs, xcoords, resultPath):
    ctpngrids = []
    if len(gridImgs) != len(xcoords):
        print 'length gridImg != length xcoords: warning!!!!!!!!!!!!!!!!!!!!!!!!!!'
    for lineIndex, gridImg in enumerate(gridImgs):
        linegrid = []
        # cv2.imshow('grid line',gridImg)
        xcoord = xcoords[lineIndex]
        if xcoord == None:
            xcoord = []
        w = gridImg.shape[1]
        h = gridImg.shape[0]
        xcoord.append(w)
        xcoord.insert(0,0)
        for i in range(0,len(xcoord)-1):
            if xcoord[i+1]-xcoord[i] < 10:
                continue
            img = np.zeros((xcoord[i+1]-xcoord[i],h),np.uint8)
            img = gridImg[0:h,xcoord[i]:xcoord[i+1]]
            # cv2.imshow('grid',img)
            # key = cv2.waitKey(0)
            # if key>0:
            #     print ' '
            cv2.imwrite(os.path.join(resultPath, str(lineIndex) + '-'+str(i)+'.jpg'), img)
            linegrid.append(img)
        ctpngrids.append(linegrid)

    return ctpngrids


def ctpnGrid(gridImgs, xcoords, reultPath):
    ctpngrids = []
    if len(gridImgs)!=len(xcoords):
        print 'length gridImg != length xcoords: warning!!!!!!!!!!!!!!!!!!!!!!!!!!'
    for lineIndex, gridImg in enumerate(gridImgs):
        xcoord = xcoords[lineIndex]
        if xcoord == None:
            xcoord = []
        w = gridImg.shape[1]
        h = gridImg.shape[0]
        text_rects = ctpnport.getCharBlock(ctpn_sess, ctpn_net, gridImg)
        linegrid = []
        gridIndex = 0
        # ctpn检测的box中，用xcoords的分界线划分成真正的单元格
        text_rects.sort()

        # # debug
        tmpGridImg = copy.copy(gridImg)
        for box in text_rects:
            minx = max(box[0][0] - 5, 0)
            maxx = min(box[1][0] + 5, w)
            miny = max(box[0][1] - 3, 0)
            maxy = min(box[3][1] + 3, h)
            cv2.rectangle(tmpGridImg,(minx,miny),(maxx,maxy),(0,255,0),2)

        for box in text_rects:
            minx = max(box[0][0]-5,0)
            maxx = min(box[1][0]+5,w)
            miny = max(box[0][1]-3,0)
            maxy = min(box[3][1]+3,h)
            xgrid = []
            xgrid.append(minx)
            xgrid.append(maxx)
            for x in xcoord:
                if x > minx and x < maxx:
                    xgrid.append(x)
            xgrid.sort()
            for i in range(len(xgrid)-1):
                x1 = xgrid[i]
                x2 = xgrid[i+1]
                if x2-x1 < 10:
                    continue
                grid = np.zeros((x2 - x1, maxy - miny), np.uint8)
                grid = gridImg[miny:maxy, x1:x2]
                # 扩充边界
                grid = preprocess.expand(grid,ratio=0.2)
                linegrid.append(grid)
                gridIndex+=1
                # # debug
                # cv2.imshow('grid',grid)
                # key = cv2.waitKey(0)
                # if key>0:
                cv2.imwrite(os.path.join(reultPath, str(lineIndex) + str(gridIndex)+'.jpg'), grid)
                #     continue
        ctpngrids.append(linegrid)
    return ctpngrids


if __name__=='__main__':
    #init pathes
    data_root_dir = os.path.join(this_dir, 'data') #数据路径
    image_dir = os.path.join(this_dir, 'data', 'image') #数据路径
    excel_dir = os.path.join(this_dir, 'data', 'excel') #数据路径
    tmp_dir = os.path.join(this_dir, 'temp') #临时文件夹

    #确定输出路径存在
    for _dir in [data_root_dir, image_dir, excel_dir, tmp_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    #清空临时文件夹和结果文件夹
    for _dir in [tmp_dir, excel_dir]:
        for temp_file in os.listdir(_dir):
            temp_file_path = os.path.join(_dir, temp_file)
            if os.path.isfile(temp_file_path):
                os.remove(temp_file_path)
            if os.path.isdir(temp_file_path):
                if len(os.listdir(temp_file_path)) == 0:
                    os.rmdir(temp_file_path)
                else:
                    for fp in os.listdir(temp_file_path):
                        os.remove(os.path.join(temp_file_path, fp))
                    os.rmdir(temp_file_path)


    # 预处理获得条形grid和竖向分界坐标
    for img_name in os.listdir(image_dir):
        gridImages, xcoords = lineDetect(os.path.join(image_dir, img_name))
        if len(gridImages) == 0:
            print('can''t find text lines in image %s.' % img_name)
            continue
        else:
            pass
            # if gridImages[0].shape[1] / gridImages[0].shape[0] > 2:
            #     gridImages, xcoords = lineDetect(os.path.join(image_dir, img_name), rotate = True)

        # ctpngrids = {[grid1_1 grid1_2...] [grid2_1...]...} 得到白底黑字BGR单元格图像
        ctpngrids = singleGrid(gridImages, xcoords, os.path.join(data_root_dir, 'ctpnGrid')) #LineGrid
        # ctpngrids = ctpnGrid(gridImages, xcoords, os.path.join(data_root_dir,'ctpnGrid')) # CTPN

        sub_folder_dir = os.path.join(tmp_dir, img_name[:-4])
        if os.path.exists(sub_folder_dir):
            if len(os.listdir(sub_folder_dir)) != 0:
                for fp in os.listdir(sub_folder_dir):
                    os.remove(os.path.join(sub_folder_dir, fp))
        else:
            os.mkdir(sub_folder_dir)

        for line_index, line in enumerate(ctpngrids):
            for grid_index, grid in enumerate(line):
                text_boxes = tableprocess.getRevisedTextLine(grid)

                if len(text_boxes) == 0:
                    continue

                box = text_boxes[0]
                if len(text_boxes) > 1:
                    minx_0, miny_0 = text_boxes[0].min(0)
                    maxx_0, maxy_0 = text_boxes[0].max(0)
                    minx_1, miny_1 = text_boxes[1].min(0)
                    maxx_1, maxy_1 = text_boxes[1].max(0)
                    #两者处于同一水平线

                    if abs(miny_0 - miny_1) < 5 and abs(maxy_0 - maxy_1) < 5:
                        box = np.array([[min(minx_0, minx_1), min(miny_0, miny_1)],
                               [max(maxx_0, maxx_1), max(maxy_0, maxy_1)],
                               [max(maxx_0, maxx_1), min(maxy_0, maxy_1)],
                               [min(maxx_0, maxx_1), max(maxy_0, maxy_1)]
                               ])
                    else:#其中u一个是虚检测，只考虑单行文本
                        subimg_0 = grid[max(miny_0, 0): min(maxy_0,grid.shape[0]), max(minx_0, 0):min(maxx_0, grid.shape[1])]
                        subimg_1 = grid[max(miny_1, 0): min(maxy_1,grid.shape[0]), max(minx_1, 0):min(maxx_1, grid.shape[1])]
                        subimg_0 = cv2.cvtColor(subimg_0, cv2.COLOR_BGR2GRAY)
                        subimg_1 = cv2.cvtColor(subimg_1, cv2.COLOR_BGR2GRAY)
                        _, binary_0 = cv2.threshold(subimg_0, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
                        _, binary_1 = cv2.threshold(subimg_1, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

                        bgd_ratio_0 = np.sum(binary_0) / (1.0 * binary_0.shape[0] * binary_0.shape[1]) #前景像素值为255
                        bgd_ratio_1 = np.sum(binary_1) / (1.0 * binary_1.shape[0] * binary_1.shape[1])

                        if bgd_ratio_0 > bgd_ratio_1:
                            box = text_boxes[1]

                #print box
                minx, miny = box.min(0)
                maxx, maxy = box.max(0)
                minx = max(minx - 4, 0)
                miny = max(miny - 2, 0)
                maxy = min(maxy + 4, grid.shape[0])
                maxx = min(maxx + 2, grid.shape[1])

                subimg = grid[miny: maxy, minx: maxx, :]

                if subimg is None:
                    continue
                if subimg.shape[0] < 1 or subimg.shape[1] < 1:
                    continue

                cur_img_path = os.path.join(sub_folder_dir, str(line_index) + '_' + str(grid_index) + '.jpg')
                cv2.imwrite(cur_img_path, subimg)

        # 权宜之计，先在caffe中识别并写入文件，然后再逐行解析，若是数字串，那么采用tesseract的识别结果
        results_text_path = os.path.join(sub_folder_dir, img_name[:-4] + '.txt')
        caffe_module.pycaffe.recoOpticalCharsInFloder(sub_folder_dir, results_text_path)

        workbook = xlsxwriter.Workbook(os.path.join(excel_dir, img_name[:-4]+'.xlsx'))
        worksheet = workbook.add_worksheet()

        with open(results_text_path, 'r') as infile:
            for line in infile.readlines():
                line = line.strip('\n')
                img_path, text_seq = line.split(' ')
                tmp_img_name, _ = img_path.split('/')[-1].split('.')
                row, col = tmp_img_name.split('_')
                count_of_number = 0
                for character in text_seq:
                    if character.isdigit():
                        count_of_number += 1
                        if count_of_number > 3:
                            break

                #数字的识别tesseract更准确一些
                if count_of_number > 3:
                    tess_num_text = tesseractport.tess_recognition_from_file_english(img_path)
                    revised_number_seq = ''
                    for character in tess_num_text:
                        if character.isdigit():
                            revised_number_seq += str(character)
                        if character == " " or character == "," or character == ".":
                            if len(revised_number_seq) > 0:
                                if revised_number_seq[-1].isdigit():
                                    revised_number_seq += "."


                    print('tesseract number: %s' % revised_number_seq)
                    worksheet.write(int(row), int(col), revised_number_seq.decode('utf-8'))
                else:
                    worksheet.write(int(row), int(col), text_seq.decode('utf-8'))
                    print(text_seq)

        workbook.close()