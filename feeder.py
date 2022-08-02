import queue
import cv2
from queue import Queue
import sys
import numpy as np
# from netmanager import KongRequest

import threading

class LoadStreams:
    def __init__(self, i=0, source=0, zed_flag=False):
        self.que = Queue(1)
        self.source = source
        self.cap = cv2.VideoCapture(int(source))
        self.zed_flag = zed_flag

    def get_frame(self, i=0, source=0):
        assert self.cap.isOpened(), f'Failed to open cam {self.source}'

        # cv2.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        # cv2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        # cv2.set(cv2.CAP_PROP_FPS, 30)
        
       
        ret, frame = self.cap.read()
        
        if not ret:
            print(f"Webcam {i} load failed")
            sys.exit()
        
        if frame is not None:
            self.que.put(frame)

        return self.que

