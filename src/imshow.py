import pyautogui
import cv2
import numpy as np 
from typing import List
from pathlib import Path
import os
from .utils import compute_winratio

WIND_MOVED = []
WIND_DESTROYS = []
WIND_NAMES = []
WIND_NAMES_BLOCKED = []

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

def reset_moved():
    global WIND_MOVED
    WIND_MOVED.clear()

def reset_names():
    global WIND_MOVED
    WIND_MOVED.clear()

def reset_destroys():
    global WIND_DESTROYS
    WIND_DESTROYS.clear()

def reset_all():
    global WIND_MOVED, WIND_DESTROYS, WIND_NAMES
    try:
        cv2.destroyAllWindows()
    except Exception as e:  # Corrected syntax
        print(e)
    WIND_MOVED.clear()
    WIND_DESTROYS.clear()
    WIND_NAMES.clear()

def destroys():
    global WIND_DESTROYS, WIND_NAMES
    if len(WIND_DESTROYS) > 0:
        for wnd in WIND_DESTROYS:
            WIND_DESTROYS.remove(wnd)
            WIND_NAMES.remove(wnd)
            if not cv2.getWindowProperty(wnd, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow(wnd)
        WIND_DESTROYS.clear()
        WIND_NAMES.clear()
        return True
    return False

def imshow(
        image, 
        winname='imshow',
        winratio=-1,
        winborder_outside=0.2,
        winmove=(0,-0.05)):
    global WIND_MOVED, SCREEN_WIDTH, SCREEN_HEIGHT, WIND_NAMES, WIND_NAMES_BLOCKED

    if winname in WIND_NAMES_BLOCKED:
        return _callback(winname)

    if winname not in WIND_NAMES:
        WIND_NAMES.append(winname)

    if isinstance(image, (str, Path)):
        assert os.path.exists(image), "File not found!"
        image = cv2.imread(image)

    img_height, img_width = image.shape[:2]

    if winratio == -1:
        screan_width=SCREEN_WIDTH-((SCREEN_WIDTH*winborder_outside)/2)
        screan_height=SCREEN_HEIGHT-((SCREEN_HEIGHT*winborder_outside)/2)
        comp_win = compute_winratio(img_width, 
                                    img_height,
                                    screan_width, 
                                    screan_height)
        image_size = np.array([img_width, img_height]) * comp_win
    else:
        image_size = np.array([img_width, img_height]) * winratio

    image_size = image_size.astype(np.int32)
    plot = image.copy()
    plot = cv2.resize(plot, image_size)
    org = np.array([SCREEN_WIDTH-image_size[0],
                    SCREEN_HEIGHT-image_size[1]]) // 2 
    
    org = org + np.array(np.array(winmove) * (SCREEN_WIDTH, SCREEN_HEIGHT))
    cv2.imshow(winname, plot)
    if winname not in WIND_MOVED:
        cv2.moveWindow(winname, *org.astype(np.int32))
        WIND_MOVED.append(winname)
    
    return _callback(winname)

class _callback:
    def __init__(self, winname):
        self.winname = winname

    def interrupt(self, waitKey, endKey='q'):
        global WIND_DESTROYS, WIND_NAMES, WIND_NAMES_BLOCKED

        if self.winname in WIND_NAMES_BLOCKED:
            return True
        
        if waitKey == 0: # image dis
            _key = cv2.waitKey(waitKey) & 0xFF
            if cv2.getWindowProperty(self.winname, cv2.WND_PROP_VISIBLE) < 1:
                WIND_NAMES.remove(self.winname)
                return True
            if _key and self.winname not in WIND_DESTROYS:
                WIND_DESTROYS.append(self.winname)
        else: # video frame
            _key = cv2.waitKey(waitKey) & 0xFF
            if cv2.getWindowProperty(self.winname, cv2.WND_PROP_VISIBLE) < 1:
                WIND_NAMES.remove(self.winname)
                return True
            if _key == ord(endKey) or _key == 27:
                WIND_DESTROYS.append(self.winname)
        return destroys()
    
def interrupt(waitKey, endKey='q'):
    global WIND_DESTROYS, WIND_NAMES
    if waitKey == 0: # image dis
        cv2.waitKey(waitKey)
        cv2.destroyAllWindows()
        WIND_DESTROYS.clear()
        WIND_NAMES.clear()
    else: # video frame
        keyborad_intr = cv2.waitKey(waitKey) & 0xFF == ord(endKey)
        wind_prop = []
        for wind in WIND_NAMES:
            prop = cv2.getWindowProperty(wind, cv2.WND_PROP_VISIBLE) < 1
            wind_prop.append(prop)
            if prop:
                WIND_NAMES.remove(wind)
        return keyborad_intr or any(wind_prop)
    return destroys()

def interrupts(waitKey):
    global WIND_DESTROYS, WIND_NAMES
    if waitKey == 0: # image dis
        cv2.waitKey(waitKey)
        cv2.destroyAllWindows()
        WIND_DESTROYS.clear()
        WIND_NAMES.clear()
    else: # video frame
        cv2.waitKey(waitKey)
        wind_prop = []
        for wind in WIND_NAMES:
            prop = cv2.getWindowProperty(wind, cv2.WND_PROP_VISIBLE) < 1
            wind_prop.append(prop)
            if prop and wind not in WIND_NAMES_BLOCKED:
                WIND_NAMES_BLOCKED.append(wind)
        return all(wind_prop)
    return destroys()

def destroyAllWindows():
    global WIND_DESTROYS, WIND_NAMES, WIND_NAMES_BLOCKED
    WIND_DESTROYS.clear()
    WIND_NAMES.clear()
    WIND_NAMES_BLOCKED.clear()
    cv2.destroyAllWindows()