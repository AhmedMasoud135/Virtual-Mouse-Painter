import cv2
import time
import HandTrackingFunctions as htf
import MouseFunctions
import numpy as np

# Camera and screen settings
wcam, hcam = 648, 488
screen_size = MouseFunctions.autopy.screen.size()

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img, results = htf.find_hands(img)
    lm_list, bbox = htf.find_positions(img, results, drawBBox=False)
    hand_types = htf.get_hand_types(results)
    fingers = htf.fingers_up([lm_list],hand_types)
    click_length, click_line = htf.find_distance(lm_list, 4, 6, img, draw=False)
    drag_length, drag_line = htf.find_distance(lm_list, 4, 12, img, draw=False)

    if len(lm_list) != 0:
        # Get finger-tip coords
        x_index, y_index = lm_list[8][1:]
        x_middle, y_middle = lm_list[12][1:]

        if fingers[1] == 1:
            # Move cursor
            MouseFunctions.move_cursor(img,x_index, y_index, (wcam, hcam), screen_size)

            # Left Click
            MouseFunctions.click_mouse(img,click_length,click_line)

            # Drag
            MouseFunctions.drag_mouse(img,drag_length,drag_line)

            if fingers[2] == 1:
                # Double Click
                MouseFunctions.double_click_mouse(img,click_length,click_line)

                if fingers[0] == 0:
                    # Scroll
                    MouseFunctions.scroll_mouse(img,x_index, y_index, x_middle, y_middle)

    cv2.imshow("Live Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
