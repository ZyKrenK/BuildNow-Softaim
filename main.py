import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab

roi_top = 0
roi_bottom = pyautogui.size().height
roi_left = 0
roi_right = pyautogui.size().width

head_image = cv2.imread('https://media.discordapp.net/attachments/912917993372143648/1128871344692531200/image.png?width=116&height=119')

while True:
    screenshot = np.array(ImageGrab.grab(bbox=(roi_left, roi_top, roi_right, roi_bottom)))

    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    gray_head = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_screenshot, gray_head, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + head_image.shape[1], top_left[1] + head_image.shape[0])

    cv2.rectangle(screenshot, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('Game', screenshot)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
