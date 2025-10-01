import cv2
import mediapipe as mp
import time
import HandTrackingFunctions
import math

cap = cv2.VideoCapture(0)

_tip_ids = [4, 8, 12, 16, 20]  # Indices of thumb, index, middle, ring, and pinky tips
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, results = HandTrackingFunctions.find_hands(img)

    all_lm_lists, bboxes = HandTrackingFunctions.find_positions_multi(
        img, results, drawBBox=False
    )
    hand_types = HandTrackingFunctions.get_hand_types(results)

    # Draw detected hand types
    if hand_types:
        cv2.putText(img," & ".join(hand_types),(10, 40),cv2.FONT_HERSHEY_PLAIN,2,(255, 0, 0),2)

    # Select mode
    mode = "first"  # or "first"

    # Get finger states
    fingers_states = HandTrackingFunctions.fingers_up(all_lm_lists, hand_types, mode=mode)

    # Normalize to list of lists
    if mode == "first":
        fingers_states = [fingers_states]

    for lm_list, fingers, hand_type in zip(all_lm_lists, fingers_states, hand_types):
        # 1. Classify gesture
        gesture_name = HandTrackingFunctions.classify_gesture(fingers)

        # 2. Draw gesture label near wrist (landmark 0)
        if lm_list:
            x0, y0 = lm_list[0][1], lm_list[0][2]
            cv2.putText(img, f"{hand_type}: {gesture_name}", (x0, y0 - 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # 3. Draw raised fingertips
        for i, is_up in enumerate(fingers):
            if is_up:
                tip_id = _tip_ids[i]
                if tip_id < len(lm_list):
                    x, y = lm_list[tip_id][1], lm_list[tip_id][2]
                    cv2.circle(img, (x, y), 20, (0, 255, 0), 3)
                    cv2.putText(img, "Up", (x - 20, y - 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        
    for lm_list in all_lm_lists:
        if len(lm_list) > 8:  # ensure both landmarks exist
            length, info = HandTrackingFunctions.find_distance(lm_list, 4, 8, img=img, draw=True)

            # Display distance value on screen
            cv2.putText(
                img,
                f"Dist: {int(length)}",
                (info[4] + 20, info[5] - 20),  # show near midpoint
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 0),2)
    

    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()           
cv2.destroyAllWindows() 