import ctypes
import numpy as np

# load library
lib = ctypes.CDLL("./Creader/libroli_fse.so")
lib.roli_fse_open.argtypes = [ctypes.c_char_p,]
lib.roli_fse_open.restype = ctypes.c_void_p
lib.roli_fse_get_width.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
lib.roli_fse_get_height.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
lib.roli_fse_get_n_frames.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
lib.roli_fse_read.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_int32]


def read_random(rolipath):
    # open the roli file
    f = lib.roli_fse_open(rolipath.encode("utf-8"))

    # pick a frame index
    n_frames = ctypes.POINTER(ctypes.c_uint64)(ctypes.c_uint64(0))
    lib.roli_fse_get_n_frames(f, n_frames)
    frame_idx = int(np.random.choice(range(n_frames.contents.value)))

    # read the frame
    width = ctypes.POINTER(ctypes.c_uint32)(ctypes.c_uint32(0))
    lib.roli_fse_get_width(f, width)
    height = ctypes.POINTER(ctypes.c_uint32)(ctypes.c_uint32(0))
    lib.roli_fse_get_height(f, height)

    img = np.zeros((int(width.contents.value), int(height.contents.value)), dtype=np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    lib.roli_fse_read(f, img, ctypes.c_int(frame_idx))

    pyimg = np.zeros(((int(width.contents.value), int(height.contents.value))))
    count = 0
    for i in range(width.contents.value):
        for j in range(height.contents.value):
            pyimg[i, j] = img[count]
            count+=1

    return pyimg, frame_idx, n_frames.contents.value