import os
import ctypes

'''
  Caution that the parameter of c++ must be char*,the reason as below: 
  The Python C API converts Python str objects into char*, and there is an implicit conversion in C++ from char*
  (actually char const*) to std::string. If the Python strings can contain null characters, you'll have to use 
  PyString_AsStringAndSize to convert, and pass around the two values (the char* and the Py_ssize_t); there's 
  an explicit conversion of these to std::string as well: std::string( pointer, length ).
'''

if __name__ ==  '__main__':
    cur_dir = os.path.dirname(__file__)
    DLL_PATH = os.path.join(cur_dir, 'lib', 'libcaffe.so' )

    pycaffe = ctypes.cdll.LoadLibrary(DLL_PATH)

    raise NotImplementedError('the function test_ocr_chinese was deprecated!')

    pycaffe.test_ocr_chinese(("/home/wz/testProjects/tableOCR/temp"),
                             ("/home/wz/DeepLearning/OCR_working_dir/caffe_ocr_for_linux/models/OCR/chinese/densenet-no-blstm"))