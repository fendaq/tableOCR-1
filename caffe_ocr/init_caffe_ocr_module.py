import os
import ctypes
cur_dir = os.path.dirname(__file__)
DLL_PATH = os.path.join(cur_dir, 'lib', 'libcaffe.so' )
pycaffe = ctypes.cdll.LoadLibrary(DLL_PATH)
pycaffe.init("/home/wz/DeepLearning/OCR_working_dir/caffe_ocr_for_linux/models/OCR/chinese/densenet-no-blstm")