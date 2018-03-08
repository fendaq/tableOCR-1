#!/usr/bin/python
# coding: utf-8

import os
import ctypes
import cv2
from glob import iglob

this_dir = os.path.dirname(__file__)

DLL_PATH = os.path.join(this_dir,'lib', 'libtesseract.so' )
TESSDATA_PREFIX = os.path.join(this_dir,'tessdata' )
english_lang = 'eng'
chinese_lang = 'chi_sim'

tesseract_chinese = ctypes.cdll.LoadLibrary(DLL_PATH)
tesseract_chinese.TessBaseAPICreate.restype = ctypes.c_uint64   #由于系统为64位,这里类型需指明为c_uint64
api_chinese = tesseract_chinese.TessBaseAPICreate()

def tess_init_chinese():
    rc = tesseract_chinese.TessBaseAPIInit3(ctypes.c_uint64(api_chinese), TESSDATA_PREFIX, chinese_lang)
    if rc:
        tesseract_chinese.TessBaseAPIDelete(ctypes.c_uint64(api_chinese))
        print('Could not initialize tesseract.\n')
        exit(3)

def tess_recognition_from_file_chinese(img_name):
    tesseract_chinese.TessBaseAPIProcessPages(
        ctypes.c_uint64(api_chinese), img_name, None, 0, None)
    tesseract_chinese.TessBaseAPIGetUTF8Text.restype = ctypes.c_uint64
    text_out = tesseract_chinese.TessBaseAPIGetUTF8Text(ctypes.c_uint64(api_chinese))
    return ctypes.string_at(text_out)

tesseract_english = ctypes.cdll.LoadLibrary(DLL_PATH)
tesseract_english.TessBaseAPICreate.restype = ctypes.c_uint64   #由于系统为64位,这里类型需指明为c_uint64
api_english = tesseract_chinese.TessBaseAPICreate()

def tess_init_english():
    rc = tesseract_english.TessBaseAPIInit3(ctypes.c_uint64(api_english), TESSDATA_PREFIX, english_lang)
    if rc:
        tesseract_english.TessBaseAPIDelete(ctypes.c_uint64(api_english))
        print('Could not initialize tesseract.\n')
        exit(3)

def tess_recognition_from_file_english(img_name):
    tesseract_english.TessBaseAPIProcessPages(
        ctypes.c_uint64(api_english), img_name, None, 0, None)
    tesseract_english.TessBaseAPIGetUTF8Text.restype = ctypes.c_uint64
    text_out = tesseract_english.TessBaseAPIGetUTF8Text(ctypes.c_uint64(api_english))
    return ctypes.string_at(text_out)

def cleanup(temp_name):
    ''' Tries to remove files by filename wildcard path. '''
    for filename in iglob(temp_name + '*' if temp_name else temp_name):
        try:
            os.remove(filename)
        except OSError:
            pass

#Deprecated
def tessRecognition(img, tmp_dir, english = True):
    img_name = os.path.join(tmp_dir, 'img.png')
    cv2.imwrite(img_name, img)
    if english:
        text_out = tess_recognition_from_file_english(img_name)
    else:
        text_out = tess_recognition_from_file_chinese(img_name)
    cleanup(img_name)
    return text_out

def tessRecognitionByFullpath(imgpath, isenglish):
    if isenglish:
        text_out = tess_recognition_from_file_english(imgpath)
    else:
        text_out = tess_recognition_from_file_chinese(imgpath)
    cleanup(imgpath)
    return text_out

# def from_file(path):
#     tesseract.TessBaseAPIProcessPages(
#         ctypes.c_uint64(api), path, None, 0, None)
#     tesseract.TessBaseAPIGetUTF8Text.restype = ctypes.c_uint64
#     text_out = tesseract.TessBaseAPIGetUTF8Text(ctypes.c_uint64(api))
#     return ctypes.string_at(text_out)

# if __name__ == '__main__':
#     image_file_path = b'/home/cvrsg/JpHu/TestGround/tesseract/build/bin/pics/1.png'
#     # result = from_file(image_file_path)
#     tessInit()
#     result = tessRecognition(image_file_path)
#     print(result)