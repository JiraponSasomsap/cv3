import pyautogui
import cv2
import numpy as np 
import pyautogui
import cv2
import numpy as np 
from typing import List

def compute_winratio(img_width, img_height, max_width, max_height):
    winratio = min(max_width / img_width, max_height / img_height)
    return winratio

class Imshow:
    def __init__(self):
        self.WIND_MOVED = False
        self.windows_name = None
        self.orig_img = None
        self.orig_shape = None
        self.img_resize_shape = None
        self.callback = None
        self.windows_ratio = 0.5
        self.windows_name = 'show'
        self._callback = Callback()

    def addCallback(self, callback):
        self.callback = callback
        return self

    def imshow(self, 
               image, 
               windows_ratio=0.5, 
               windows_name='show'):
        
        self.windows_name = windows_name
        self.orig_img = image
        self.orig_shape = image.shape
        screen_width, screen_height = pyautogui.size()   
        img_height, img_width, _ = image.shape 
        image_size = np.array([img_width, img_height]) * windows_ratio
        image_size = image_size.astype(np.int32)
        plot = image.copy()
        plot = cv2.resize(plot, image_size)
        self.img_resize_shape = plot.shape
        org = np.array([screen_width-image_size[0],
                        screen_height-image_size[1]]) / 2 
        org = org.astype(np.int32)
        cv2.imshow(windows_name, plot)
        if not self.WIND_MOVED:
            cv2.moveWindow(windows_name, *org)
            self.WIND_MOVED = True

        if self.callback is not None:
            self.callback(self)

    def imshow_getcoords(self, 
                        image, 
                        windows_ratio=0.5, 
                        windows_name='show'):
        self.addCallback(self._callback.get_coords)
        self.imshow(image, 
                    windows_ratio=windows_ratio, 
                    windows_name=windows_name)
    
    @property
    def coords(self):
        if len(self._callback.points) == 0:
            return np.array([])
        return np.array(self._callback.points) / [self._callback.width, self._callback.height]

    def destroyAllWindows(self):
        cv2.destroyAllWindows()
    
    def interrupt(self, waitKey=1, key='q'):
        _key = cv2.waitKey(waitKey) & 0xFF  # Read key once and apply bitmask

        if _key == ord(key) or _key == 27 or cv2.getWindowProperty(self.windows_name, cv2.WND_PROP_VISIBLE) < 1:
            return True
        return False

class Callback:
    def __init__(self):
        self.height = None
        self.width = None
        self.points = []

    def get_coords(self, instance:Imshow):
        self.height, self.width, _ = instance.img_resize_shape
        cv2.setMouseCallback(instance.windows_name, self.__event_manager)
    
    @property
    def coords(self):
        if len(self.points) == 0:
            return np.array([])
        return np.array(self.points) / [self.width, self.height]

    def __event_manager(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
        if event == cv2.EVENT_RBUTTONDOWN:
            self.points.pop()



class Implot:
    @staticmethod
    def frame_count(image,
                    frame_count, 
                    org = (10,10),
                    font_color = (0,0,0),
                    background_color = (255,255,255),
                    fontFace = cv2.FONT_HERSHEY_COMPLEX,
                    fontScale = 0.6,
                    thickness = 1):
        plot = image.copy()
        org = np.array(org, dtype=np.int32)
        text = f'{frame_count} frame'
        (text_width, text_height), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
        plot = cv2.rectangle(plot,
                             org, 
                             org+[text_width+(baseline*2), text_height+(baseline*2)], 
                             background_color, 
                             cv2.FILLED)
        plot = cv2.putText(plot, 
                           text, 
                           org+[0, text_height+baseline], 
                           fontFace, 
                           fontScale, 
                           font_color,
                           thickness)
        return plot
    
    @staticmethod
    def imshow_label(image, 
                   boxes:List[List[int]], 
                   text_list:List[List[str]], 
                   label_color = (255,255,0),
                   label_backgound=True,
                   label_thickness=1,
                   fontFace = cv2.FONT_HERSHEY_COMPLEX,
                   fontScale = 1,
                   text_thickness = 1,
                   text_color = (0,0,0),
                ):
        plot = image.copy()
        for ibox, box in enumerate(boxes):
            box = np.array(box, dtype=np.int32)
            plot = cv2.rectangle(plot, box[:2], box[2:], label_color, label_thickness)
            label_bg_coords = box[:2].copy()
            label_text_coords = box[:2].copy()

            for text in text_list[ibox]:
                (text_width, text_height), baseline = cv2.getTextSize(text, 
                                                                      fontFace, 
                                                                      fontScale, 
                                                                      text_thickness,)
                if label_backgound:
                    plot = cv2.rectangle(plot, 
                                        label_bg_coords, 
                                        label_bg_coords+[text_width+(baseline*2), text_height+(baseline*2)],
                                        label_color,
                                        cv2.FILLED)
                    label_bg_coords = label_bg_coords+[0, text_height+(baseline*2)]

                plot = cv2.putText(plot,
                                   text,
                                   label_text_coords+[baseline, text_height+baseline],
                                   fontFace,
                                   fontScale,
                                   text_color,
                                   text_thickness)
                label_text_coords = label_text_coords+[0, text_height+(baseline*2)]
        return plot