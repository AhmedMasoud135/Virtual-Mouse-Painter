import cv2
import mediapipe as mp
import time
import HandTrackingFunctions as htf
import numpy as np
import os
import autopy

wcam, hcam = 648, 488
wSCR, hSCR = autopy.screen.size()
frameR = 150
smoothing = 8
plocx, plocy = 0, 0
clocx, clocy = 0, 0

# Click settings
click_state = False
click_time = 0
click_cooldown = 0.7  # 0.5 second cooldown between clicks
click_threshold = 35  # Distance threshold for click

# Drag settings
drag_threshold = 35  # Distance threshold for drag detection
drag_state = False
drag_start_time = 0
drag_hold_time = 0.3  # Time to hold gesture before drag activates
gesture_stable_time = 0.1  # Time gesture must be stable

# Previous gesture state for stability checking
prev_drag_distance = 0
gesture_start_time = 0


cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    current_time = time.time()

    # Find hand landmarks
    img, results = htf.find_hands(img)

    lm_list, bbox = htf.find_positions(
        img, results, drawBBox=False
    )

    # Get hand types [Left, Right]
    hand_types = htf.get_hand_types(results)


    # Get the tip of the index finger and middle finger
    if len(lm_list)!=0:
        x_index, y_index = lm_list[8][1:]           # tip of index finger
        x_middle, y_middle = lm_list[12][1:]          # tip of middle finger

        # Convert coordinates to screen resolution     {np.interp(value, (old_min, old_max), (new_min, new_max))}
        x3 = np.interp(x_index, (frameR, wcam-frameR), (0, wSCR))
        y3 = np.interp(y_index, (frameR, hcam-frameR), (0, hSCR))

        # Smoothen values
        clocx = plocx + (x3 - plocx) / smoothing
        clocy = plocy + (y3 - plocy) / smoothing

    
    # Find fingers up
    fingers = htf.fingers_up([lm_list],hand_types)

    # Frame Reduction
    cv2.rectangle(img, (frameR, frameR), (wcam - frameR, hcam - frameR), (255, 0, 255), 2)

    # If index finger is up
    if fingers[1] == 1 and fingers[2] == 0:
        # Move mouse
        autopy.mouse.move(clocx, clocy)
        cv2.circle(img, (x_index, y_index), 15, (255, 0, 255), cv2.FILLED)
        plocx, plocy = clocx, clocy

        # Find distance between fingers [4, 6]
        click_length, click_line = htf.find_distance(lm_list, 4, 6, img, draw=False)
        #cv2.putText(img, f" {length}", (50,50),
        #               cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # Click mouse if distance short and click cooldown is over
        if click_length < 37:
            if not click_state and (current_time - click_time) > click_cooldown:
                cv2.circle(img, (click_line[4], click_line[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click(autopy.mouse.Button.LEFT)
                click_state = True
                click_time = current_time

        else:
            if click_state:
                click_state = False

    drag_length, drag_line = htf.find_distance(lm_list, 4, 8, img, draw=False)
    #print(drag_length)
    
    # Check if drag gesture is detected
    drag_gesture_detected = drag_length < drag_threshold and drag_length > 0
    
    # Check gesture stability
    distance_stable = abs(drag_length - prev_drag_distance) < 10  # Distance change threshold
    
    if drag_gesture_detected:
        if not drag_state:
            # Start tracking potential drag gesture
            if gesture_start_time == 0:
                gesture_start_time = current_time
            elif distance_stable and (current_time - gesture_start_time) > drag_hold_time:
                # Gesture is stable for required time, initiate drag
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)  # Press and hold left button
                drag_state = True
                drag_start_time = current_time
                cv2.putText(img, "DRAG ON", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        else:
            # Drag in progress
            autopy.mouse.move(clocx, clocy)

            cv2.putText(img, "DRAGGING", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.circle(img, (drag_line[4], drag_line[5]), 15, (0, 255, 0), cv2.FILLED)
    else:
        # Gesture not detected or lost
        gesture_start_time = 0  # Reset gesture timer
        if drag_state:
            # Release drag
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)  # Release left button
            drag_state = False
            cv2.putText(img, "DRAG OFF", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    
    # Update previous distance for stability checking
    prev_drag_distance = drag_length
    
    # Visual feedback for drag detection
    if drag_gesture_detected and not drag_state:
        remaining_time = drag_hold_time - (current_time - gesture_start_time) if gesture_start_time > 0 else drag_hold_time
        cv2.putText(img, f"Hold: {remaining_time:.1f}s", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        cv2.circle(img, (drag_line[4], drag_line[5]), 15, (255, 255, 0), cv2.FILLED)

    else:
        # No hand detected, reset all states
        if drag_state:
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
            drag_state = False
        gesture_start_time = 0
        click_state = False    
        

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()           
cv2.destroyAllWindows() 