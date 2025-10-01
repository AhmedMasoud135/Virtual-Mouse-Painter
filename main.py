import cv2
import time

from seaborn import color_palette
import HandTrackingFunctions as htf
import autopy
import PainterFunctions as pf
import MouseFunctions
import numpy as np

# ----------------- Configuration -----------------
wcam, hcam = 648, 488
screen_size = MouseFunctions.autopy.screen.size()
screen_w, screen_h = int(screen_size[0]), int(screen_size[1])

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

# From helpers
mode = "MOUSE"
draw_color = (255, 0, 255)  # default purple
brush_thickness = 7
eraser_thickness = 50
img_canvas = np.zeros((screen_h, screen_w, 3), np.uint8)
header_height = 150
prev_loc = {'x': 0, 'y': 0}

# Initialize screen overlay
pf.setup_screen_overlay(screen_w, screen_h)


# Setup camera window
cv2.namedWindow("Hand Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Control", 640, 480)


# ----------------- Main Loop -----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Hand detection
    img, results = htf.find_hands(img)
    lm_list, bbox = htf.find_positions(img, results, drawBBox=False)
    hand_types = htf.get_hand_types(results)
    fingers = htf.fingers_up([lm_list], hand_types)
    click_length, click_line = htf.find_distance(lm_list, 4, 6, img, draw=False)
    drag_length, drag_line = htf.find_distance(lm_list, 4, 12, img, draw=False)

    # ---------------- Selection Panel ----------------
    # Draw selection panel
    pf.draw_selection_panel(img)

    if len(lm_list) != 0:
        x_index, y_index = lm_list[8][1:]
        x_middle, y_middle = lm_list[12][1:]
        
        # Selection gesture (index up + middle up)
        if fingers[1] == 1 and fingers[2] == 1:
            if pf.check_selection_click(x_index, y_index):

                # Get selected mode and color
                mode = pf.mode
                draw_color = pf.draw_color

                # Show/hide overlay based on mode
                if mode == "PAINT":
                    pf.show_screen_overlay()
                    print("🎨 PAINT MODE - Screen drawing ENABLED")
                else:
                    pf.hide_screen_overlay()
                    print("🖱️  MOUSE MODE - Screen drawing DISABLED")


        # ---------------- Mode Logic ----------------
        if mode == "MOUSE":
            pf.handle_mouse_mode(img, lm_list, fingers, click_length, click_line, drag_length, drag_line)


        elif mode == "PAINT":
            prev_loc = pf.handle_screen_drawing(
                lm_list, fingers, draw_color, prev_loc,
                wcam, hcam, screen_w, screen_h
            )


    # Add visual indicators
    mode_color = (0, 255, 0) if mode == "PAINT" else (255, 255, 0)
    cv2.putText(img, f"MODE: {mode}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
    
    # Show drawing status
    if mode == "PAINT":
        status = "DRAWING ON SCREEN" if pf.overlay_active else "SCREEN OVERLAY HIDDEN"
        cv2.putText(img, status, (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the camera feed
    cv2.imshow("Hand Control", img)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('c'):
        pf.clear_screen_drawings()
        print("🧹 Screen cleared!")
    elif key == ord('s'):  # Save screenshot
        import pyautogui
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(f"screen_drawing_{int(time.time())}.png")
            print("📷 Screen saved!")
        except:
            print("❌ Could not save screenshot")

# Cleanup
print("Cleaning up...")
pf.close_screen_overlay()
cap.release()
cv2.destroyAllWindows()
print("Done!")