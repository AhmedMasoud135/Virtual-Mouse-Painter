import time
import numpy as np
import autopy
import cv2
import pyautogui

# Global smoothing and frame reduction settings
SMOOTHING = 8
FRAME_R = 150
CLICK_THRESHOLD = 35      # Distance for click (thumb-index)
CLICK_HOLD_TIME = 0.1     # Seconds to hold pinch before clicking
DRAG_THRESHOLD = 35       # Distance for drag (thumb-index)
DRAG_HOLD_TIME = 0.5      # Seconds to hold pinch before dragging
DOUBLE_CLICK_MAX_INTERVAL = 0.5  # max seconds between two clicks
SCROLL_SMOOTHING   = 4      # Higher = smoother but less responsive
SCROLL_THRESHOLD   = 2      # Minimum pixel movement to trigger scroll
SCROLL_SPEED       = 100    # Multiplier for scroll amount per movement

# Internal state
prev_loc = {'x': 0, 'y': 0}
click_state = {'timer_started': False, 'start_time': 0, 'clicked': False}
drag_state = {'timer_started': False, 'start_time': 0, 'dragging': False}
double_click_state = {'last_click_time': 0.0}
scroll_state = {'prev_y': None, 'last_time': 0.0}
    
def move_cursor(img, x_raw, y_raw, cam_size, screen_size,FRAME_R=FRAME_R, SMOOTHING=SMOOTHING):
    """
    Move the mouse cursor smoothly to mapped screen coordinates.

    Args:
        x_raw (int): Raw x from camera.
        y_raw (int): Raw y from camera.
        cam_size (tuple): (width, height) of camera frame.
        screen_size (tuple): (width, height) of display.
    """
    global prev_loc
    wcam, hcam = cam_size
    wSCR, hSCR = screen_size

    # Frame Reduction
    cv2.rectangle(img, (FRAME_R, FRAME_R), (wcam - FRAME_R, hcam - FRAME_R), (255, 0, 255), 2)

    # Map within a reduced frame
    x_mapped = np.interp(x_raw, (FRAME_R, wcam - FRAME_R), (0, wSCR))
    y_mapped = np.interp(y_raw, (FRAME_R, hcam - FRAME_R), (0, hSCR))

    # Smooth movement
    x_smooth = prev_loc['x'] + (x_mapped - prev_loc['x']) / SMOOTHING
    y_smooth = prev_loc['y'] + (y_mapped - prev_loc['y']) / SMOOTHING

    autopy.mouse.move(x_smooth, y_smooth)
    prev_loc['x'], prev_loc['y'] = x_smooth, y_smooth
    cv2.circle(img, (x_raw, y_raw), 15, (255, 0, 255), cv2.FILLED)


def click_mouse(img, Click_distance,Click_line, button=autopy.mouse.Button.LEFT,CLICK_THRESHOLD=CLICK_THRESHOLD,CLICK_HOLD_TIME=CLICK_HOLD_TIME):
    """
    Perform a click when pinch (thumb-index) is held below threshold.

    Args:
        distance (float): Current distance between thumb and index tips.
        button: autopy mouse button to click (default LEFT).
    """
    global click_state
    now = time.time()

    if Click_distance < CLICK_THRESHOLD:
        if not click_state['timer_started']:
            click_state['timer_started'] = True
            click_state['start_time'] = now
        elif not click_state['clicked'] and (now - click_state['start_time']) >= CLICK_HOLD_TIME:
            autopy.mouse.click(button)
            cv2.circle(img, (Click_line[4], Click_line[5]), 15, (0, 255, 0), cv2.FILLED)
            click_state['clicked'] = True
    else:
        # Reset when fingers open
        click_state['timer_started'] = False
        click_state['clicked'] = False


def drag_mouse(img,Drag_distance,Drag_line, button=autopy.mouse.Button.LEFT):
    """
    Press, hold, and release drag when pinch (thumb-index) is held then released.

    Args:
        distance (float): Current distance between thumb and middle tips.
        button: autopy mouse button to use for drag (default LEFT).
    """
    global drag_state
    now = time.time()

    if Drag_distance < DRAG_THRESHOLD:
        if not drag_state['timer_started']:
            drag_state['timer_started'] = True
            drag_state['start_time'] = now
        elif not drag_state['dragging'] and (now - drag_state['start_time']) >= DRAG_HOLD_TIME:
            autopy.mouse.toggle(button, True)   # Press and hold
            cv2.circle(img, (Drag_line[4], Drag_line[5]), 15, (0, 255, 255), cv2.FILLED)
            drag_state['dragging'] = True
    else:
        if drag_state['dragging']:
            autopy.mouse.toggle(button, False)  # Release
        # Reset after release
        drag_state['timer_started'] = False
        drag_state['dragging'] = False


def double_click_mouse(img, Click_distance, Click_line,button=autopy.mouse.Button.LEFT,threshold=CLICK_THRESHOLD,hold_time=CLICK_HOLD_TIME,max_interval=DOUBLE_CLICK_MAX_INTERVAL):
    """
    Performs a double-click when two pinch-clicks occur within `max_interval` seconds.
    Uses the same pinch gesture as click_mouse.

    Args:
        img (ndarray):       The frame for drawing feedback.
        distance (float):    Distance between thumb and index tips.
        click_line (tuple):  Coordinates from find_distance.
        button:              autopy mouse button to use.
        threshold (int):     Pinch distance threshold.
        hold_time (float):   Seconds to hold pinch before first click.
        max_interval (float):Max time between two clicks.
    """
    global double_click_state
    now = time.time()

    # First, attempt a single click with hold-time
    click_mouse(img, Click_distance, Click_line, button, threshold, hold_time)

    # If a click just occurred 
    if click_state['clicked']:
        # If previous click was within interval, fire double-click
        if now - double_click_state['last_click_time'] <= max_interval:
            autopy.mouse.click(button)
            cv2.circle(img, (Click_line[4], Click_line[5]), 20, (0, 165, 255), cv2.FILLED)
            # Reset to avoid triple-click
            double_click_state['last_click_time'] = 0.0
        else:
            # Record this click time for double-click detection
            double_click_state['last_click_time'] = now


def scroll_mouse(img, x_index, y_index, x_middle, y_middle, smoothing=SCROLL_SMOOTHING, threshold=SCROLL_THRESHOLD, speed=SCROLL_SPEED):
    """
    Scrolls vertically when index and middle fingers are up.
    The scroll amount is determined by the average vertical movement
    of the two fingertips between frames.

    Args:
        img (ndarray):          Frame for optional feedback (not used here).
        y_index (int):          Current y of index fingertip.
        y_middle (int):         Current y of middle fingertip.
        smoothing (int):        Factor to smooth out rapid jitter.
        threshold (int):        Min pixel delta to trigger scroll.
        speed (float):          Scroll units per pixel movement.
    """
    global scroll_state
    now = time.time()
    avg_y = (y_index + y_middle) / 2
    avg_x = (x_index + x_middle) / 2

    # Initialize previous y on first call
    if scroll_state['prev_y'] is None:
        scroll_state['prev_y'] = avg_y
        return

    # Compute movement delta
    delta = scroll_state['prev_y'] - avg_y
    # Smooth jitter by scaling down
    delta_smooth = delta / smoothing

    # Only scroll if movement is significant
    if abs(delta_smooth) >= threshold / smoothing:
        # autopy.scroll uses (horizontal, vertical)
        pyautogui.scroll(int(delta_smooth * speed))
        cv2.circle(img, (int(avg_x), int(avg_y)), 20, (120, 165, 255), cv2.FILLED)
        scroll_state['last_time'] = now

    scroll_state['prev_y'] = avg_y