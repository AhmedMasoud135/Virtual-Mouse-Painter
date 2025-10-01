# Virtual Painter & Mouse - Functional Version (Non-OOP)
import cv2
import time
import HandTrackingFunctions as htf
import MouseFunctions
import numpy as np
import tkinter as tk
from tkinter import Canvas
import threading
import time

# ----------------- Configuration -----------------
wcam, hcam = 640, 480           # Camera resolution
FRAME_R = 150                   # Margin for reduced frame
SMOOTHING = 7                   # Smoothing factor
screen_size = screen_w, screen_h = MouseFunctions.autopy.screen.size()
screen_w, screen_h = int(screen_w), int(screen_h)

# Canvas and drawing settings
mode = "MOUSE"  # Default mode: MOUSE or PAINT

draw_color = (255, 0, 255)  # Default purple
brush_thickness = 7
eraser_thickness = 50

img_canvas = np.zeros((screen_h, screen_w, 3), np.uint8)
header_height = 150
prev_loc = {'x': 0, 'y': 0}

# screen overlay
overlay_root = None
overlay_canvas = None
overlay_active = False
overlay_thread = None


# Selection panel setup
mode_selector = {
    "MOUSE": (25, 50, 125, 100),   # x1, y1, x2, y2
    "PAINT": (155, 50, 255, 100)
}


color_palette = {
    "RED": ((305, 50, 355, 100), (0, 0, 255)),
    "GREEN": ((375, 50, 425, 100), (0, 255, 0)),
    "BLUE": ((445, 50, 495, 100), (255, 0, 0)),
    "YELLOW": ((515, 50, 565, 100), (0, 255, 255)),
    "PINK": ((585, 50, 635, 100), (255, 0, 255)),
    "ERASER": ((155, 110, 255, 130), (0, 0, 0))
}



# ----------------- Screen Overlay Functions -----------------
def setup_screen_overlay(screen_w, screen_h):
    """Initialize screen overlay - call once at startup"""
    global overlay_root, overlay_canvas, overlay_thread
    
    def create_overlay():
        global overlay_root, overlay_canvas
        overlay_root = tk.Tk()
        
        # Make transparent and always on top
        overlay_root.attributes('-alpha', 0.7)
        overlay_root.attributes('-topmost', True)
        overlay_root.attributes('-transparentcolor', 'black')
        
        # Fullscreen, no decorations
        overlay_root.overrideredirect(True)
        overlay_root.geometry(f"{screen_w}x{screen_h}+0+0")
        
        # Create transparent canvas
        overlay_canvas = Canvas(
            overlay_root, 
            width=screen_w, 
            height=screen_h,
            bg='black',
            highlightthickness=0
        )
        overlay_canvas.pack()
        
        overlay_root.withdraw()  # Start hidden
        overlay_root.mainloop()
    
    # Start overlay in background thread
    overlay_thread = threading.Thread(target=create_overlay, daemon=True)
    overlay_thread.start()
    time.sleep(0.5)  # Wait for initialization

def show_screen_overlay():
    """Show the screen overlay for drawing"""
    global overlay_root, overlay_active
    overlay_active = True
    if overlay_root:
        try:
            overlay_root.deiconify()
            overlay_root.attributes('-topmost', True)
        except:
            pass

def hide_screen_overlay():
    """Hide the screen overlay"""
    global overlay_root, overlay_active
    overlay_active = False
    if overlay_root:
        try:
            overlay_root.withdraw()
        except:
            pass

def draw_on_screen(x1, y1, x2, y2, color=(255, 0, 255), width=7):
    """Draw line on screen overlay"""
    global overlay_canvas, overlay_active, overlay_root
    if overlay_canvas and overlay_active:
        try:
            # Convert BGR to hex
            hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
            
            overlay_canvas.create_line(
                x1, y1, x2, y2,
                fill=hex_color,
                width=width,
                capstyle=tk.ROUND,
                smooth=True
            )
            overlay_root.update()
        except:
            pass

def clear_screen_drawings():
    """Clear all drawings from screen"""
    global overlay_canvas, overlay_root
    if overlay_canvas:
        try:
            overlay_canvas.delete("all")
            overlay_root.update()
        except:
            pass

def close_screen_overlay():
    """Close screen overlay completely"""
    global overlay_root, overlay_active
    overlay_active = False
    if overlay_root:
        try:
            overlay_root.quit()
            overlay_root.destroy()
        except:
            pass

def handle_screen_drawing(lm_list, fingers, draw_color, prev_loc, wcam, hcam, screen_w, screen_h):
    """Handle drawing on screen overlay"""
    global overlay_active
    if not lm_list or not overlay_active:
        return prev_loc
    
    # Get finger position
    x_raw, y_raw = lm_list[8][1], lm_list[8][2]
    
    # Skip header area
    if y_raw < 150:
        return prev_loc
    
    # Map camera to screen coordinates
    x_screen = np.interp(x_raw, (150, wcam - 150), (0, screen_w))
    y_screen = np.interp(y_raw, (150, hcam - 150), (0, screen_h))
    
    # Smooth movement
    SMOOTHING = 7
    x_smooth = prev_loc.get('x', x_screen) + (x_screen - prev_loc.get('x', x_screen)) / SMOOTHING
    y_smooth = prev_loc.get('y', y_screen) + (y_screen - prev_loc.get('y', y_screen)) / SMOOTHING
    
    prev_loc.update({'x': x_smooth, 'y': y_smooth})
    
    # Drawing gesture (index up, middle down)
    if fingers[1] == 1 and fingers[2] == 0:
        if 'last_x' in prev_loc and 'last_y' in prev_loc:
            # Draw line from last position to current
            thickness = 50 if draw_color == (0, 0, 0) else 7  # Thicker eraser
            draw_on_screen(
                prev_loc['last_x'], prev_loc['last_y'],
                x_smooth, y_smooth,
                color=draw_color,
                width=thickness
            )
        
        prev_loc['last_x'] = x_smooth
        prev_loc['last_y'] = y_smooth
    else:
        # Reset when not drawing
        prev_loc.pop('last_x', None)
        prev_loc.pop('last_y', None)
    
    return prev_loc



# ----------------- Functions -----------------
def draw_selection_panel(img):
    """Draw the selection panel on the image"""
    global mode, draw_color
    
    # Draw header background
    cv2.rectangle(img, (0, 0), (wcam, header_height), (50, 50, 50), cv2.FILLED)
    
    # Draw mode selector
    for mode_name, coords in mode_selector.items():
        x1, y1, x2, y2 = coords
        color = (0, 255, 0) if mode == mode_name else (100, 100, 100)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, mode_name, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    
    # Draw color palette (only when in PAINT mode)
    if mode == "PAINT":
        for color_name, (coords, color_bgr) in color_palette.items():
            x1, y1, x2, y2 = coords
            if color_name == "ERASER":
                cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), cv2.FILLED)
                cv2.putText(img, "ERASE", (x1 + 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, cv2.FILLED)
    
        # Show current drawing color indicator
        cv2.circle(img, (25, 25), 15, draw_color, cv2.FILLED)



def check_selection_click(x, y):
    """Check if click is in selection panel and handle selection"""
    global mode, draw_color
    
    if y < header_height:  # Click in header area
        # Check mode selector
        for mode_name, coords in mode_selector.items():
            x1, y1, x2, y2 = coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                mode = mode_name
                print(f"Mode changed to: {mode}")
                return True
        
        # Check color palette (only in PAINT mode)
        if mode == "PAINT":
            for color_name, (coords, color_bgr) in color_palette.items():
                x1, y1, x2, y2 = coords
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if color_name == "ERASER":
                        draw_color = (0, 0, 0)
                    else:
                        draw_color = color_bgr
                    print(f"Color changed to: {color_name}")
                    return True
    return False




def handle_mouse_mode(img, lm_list, fingers, click_length, click_line, drag_length, drag_line):
    """Handle mouse functionality"""
    if len(lm_list) != 0:
        x_index, y_index = lm_list[8][1:]
        x_middle, y_middle = lm_list[12][1:]
        
        # Check if clicking in selection area
        # if check_selection_click(x_index, y_index):
        #     return
        
        if fingers[1] == 1:
            # Move cursor
            MouseFunctions.move_cursor(img, x_index, y_index, (wcam, hcam), screen_size)
            
            # Left Click
            MouseFunctions.click_mouse(img, click_length, click_line)
            
            # Drag
            MouseFunctions.drag_mouse(img, drag_length, drag_line)
            
            if fingers[2] == 1:
                # Double Click
                MouseFunctions.double_click_mouse(img, click_length, click_line)
                
                if fingers[0] == 0:
                    # Scroll
                    MouseFunctions.scroll_mouse(img, x_index, y_index, x_middle, y_middle)



def handle_paint_mode(img, lm_list, fingers, img_canvas= img_canvas, prev_loc = prev_loc, FRAME_R =FRAME_R,
                      wcam = wcam, hcam = hcam, screen_w = screen_w, screen_h = screen_h,
                      draw_color = draw_color, brush_thickness = brush_thickness, eraser_thickness = eraser_thickness, SMOOTHING =SMOOTHING):
    """
    Handles painting: maps camera coords to full-screen canvas with reduced frame,
    applies smoothing, and draws on img_canvas.
    """
    # Default last draw pos
    xp, yp = prev_loc.get('xp', 0), prev_loc.get('yp', 0)

    if not lm_list:
        return img_canvas, prev_loc, xp, yp

    # Raw index finger tip coords
    x_raw, y_raw = lm_list[8][1], lm_list[8][2]

    # If selecting panel, do nothing here
    if y_raw < FRAME_R:
        return img_canvas, prev_loc, xp, yp

    # Map raw to screen with reduced frame
    x_mapped = np.interp(x_raw, (FRAME_R, wcam - FRAME_R), (0, screen_w))
    
    y_mapped = np.interp(y_raw, (FRAME_R, hcam - FRAME_R), (0, screen_h))

    # Smooth movement
    x_smooth = prev_loc['x'] + (x_mapped - prev_loc['x']) / SMOOTHING
    y_smooth = prev_loc['y'] + (y_mapped - prev_loc['y']) / SMOOTHING
    prev_loc.update({'x': x_smooth, 'y': y_smooth})

    # Drawing logic
    # Selection gesture (index+middle up)
    if fingers[1] == 1 and fingers[2] == 1:
        # Show cursor indicator  (remapped to frame coords)
        cv2.circle(img, (int(np.interp(x_smooth, (0, screen_w), (0, wcam))),
                         int(np.interp(y_smooth, (0, screen_h), (0, hcam)))),
                   15, draw_color, cv2.FILLED)
        xp, yp = x_smooth, y_smooth

    # Drawing gesture (index up, middle down)
    elif fingers[1] == 1 and fingers[2] == 0:
        # Initialize starting point
        if xp == 0 and yp == 0:
            xp, yp = x_smooth, y_smooth

        # Choose thickness
        thickness = eraser_thickness if draw_color == (0, 0, 0) else brush_thickness

        # Draw line on full-screen canvas
        cv2.line(img_canvas, (int(xp), int(yp)), (int(x_smooth), int(y_smooth)), draw_color, thickness)
        xp, yp = x_smooth, y_smooth

    else:
        # Reset when no drawing gesture
        xp, yp = 0, 0

    # Store last draw position
    prev_loc.update({'xp': xp, 'yp': yp})

    return img_canvas, prev_loc, xp, yp


