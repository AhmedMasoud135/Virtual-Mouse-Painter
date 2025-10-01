import cv2
import mediapipe as mp
import time
import math

# Initialize MediaPipe hands once
_mp_hands = mp.solutions.hands
_hands = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
_mp_draw = mp.solutions.drawing_utils
_tip_ids = [4, 8, 12, 16, 20]  # Indices of thumb, index, middle, ring, and pinky tips



def find_hands(img, draw=True):
    """
    Detects hand landmarks in a BGR image and optionally draws them.
    
    Args:
        img (numpy.ndarray): Input BGR image.
        draw (bool): If True, draw landmarks and connections on img.
    
    Returns:
        tuple:
            img_out (numpy.ndarray): Image with landmarks drawn (if draw=True).
            results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                Raw landmark detection results for further processing.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = _hands.process(img_rgb)
    if draw and results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            _mp_draw.draw_landmarks(
                img, hand_lms, _mp_hands.HAND_CONNECTIONS
            )
    return img, results


# Extract pixel coordinates for one hand
def find_positions(img, results, hand_no=0, drawLM=True, drawBBox=True):
    """
    Extracts pixel coordinates of 21 hand landmarks for a specified hand.
    
    Args:
        img (numpy.ndarray): Input BGR image (for drawing).
        results: Output from find_hands(), containing multi_hand_landmarks.
        hand_no (int): Index of the hand to process (0 or 1).
        draw (bool): If True, draw small circles at each landmark.
    
    Returns:
        tuple:
            lm_list (list of [id, x, y]): Pixel coordinates of each landmark.
            bbox (tuple): (xmin, ymin, xmax, ymax) bounding box of the hand.
    """
    lm_list, x_list, y_list = [], [], []
    if not results.multi_hand_landmarks or hand_no >= len(results.multi_hand_landmarks):
        return [], ()
    hand = results.multi_hand_landmarks[hand_no]
    h, w, _ = img.shape
    for idx, lm in enumerate(hand.landmark):
        x, y = int(lm.x * w), int(lm.y * h)  # put into pixel scale
        lm_list.append([idx, x, y])
        x_list.append(x)
        y_list.append(y)
        if drawLM:
            cv2.circle(img, (x, y), 5, (255, 0, 255), cv2.FILLED)
    xmin, xmax = min(x_list), max(x_list)
    ymin, ymax = min(y_list), max(y_list)
    if drawBBox:
        cv2.rectangle(img, (xmin - 20, ymin - 20),
                      (xmax + 20, ymax + 20), (0, 255, 0), 2)
    return lm_list, (xmin, ymin, xmax, ymax)



# Extract pixel coordinates for all hands
def find_positions_multi(img, results, drawLM=True, drawBBox=True):
    """
    Extracts pixel coordinates for all detected hands.

    Returns:
        all_lm_lists: List of lm_list for each hand.
        all_bboxes:   List of bbox tuples for each hand.
    """
    all_lm_lists = []
    all_bboxes  = []

    if not results.multi_hand_landmarks:
        return all_lm_lists, all_bboxes

    h, w, _ = img.shape
    for hand in results.multi_hand_landmarks:
        lm_list, x_list, y_list = [], [], []
        for idx, lm in enumerate(hand.landmark):
            px, py = int(lm.x * w), int(lm.y * h)
            lm_list.append([idx, px, py])
            x_list.append(px)
            y_list.append(py)
            if drawLM:
                cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)
        bbox = (xmin, ymin, xmax, ymax)
        if drawBBox:
            cv2.rectangle(img, (xmin - 20, ymin - 20),
                          (xmax + 20, ymax + 20), (0, 255, 0), 2)

        all_lm_lists.append(lm_list)
        all_bboxes.append(bbox)

    return all_lm_lists, all_bboxes


# Determine raised fingers for one hand
def fingers_up_single(lm_list, hand_type="Right"):
    """
    Determines which fingers are raised for a single hand.
    Returns [thumb, index, middle, ring, pinky].
    """
    if not lm_list or len(lm_list) <= max(_tip_ids):
        return [0, 0, 0, 0, 0]

    fingers = []

    # Thumb: compare x based on hand side
    if hand_type == "Left":
        fingers.append(1 if lm_list[_tip_ids[0]][1] > lm_list[_tip_ids[0] - 1][1] else 0)
    else:
        fingers.append(1 if lm_list[_tip_ids[0]][1] < lm_list[_tip_ids[0] - 1][1] else 0)

    # Other fingers: tip y < pip y → finger is up
    for i in range(1, 5):
        tip_y = lm_list[_tip_ids[i]][2]
        pip_y = lm_list[_tip_ids[i] - 2][2]
        fingers.append(1 if tip_y < pip_y else 0)

    return fingers


# Determine raised fingers for all hands
def fingers_up(all_hands_lm, hand_types=None, mode="first"):
    """
    Determines raised fingers for detected hands.
    mode="first" → returns one list,
    mode="all"   → returns list of lists.
    """
    # No hands detected
    if not all_hands_lm:
        return [] if mode == "all" else [0, 0, 0, 0, 0]

    n_hands = len(all_hands_lm)

    # Default all to "Right"
    if hand_types is None or len(hand_types) != n_hands:
        hand_types = ["Right"] * n_hands

    # FIRST mode: only first hand
    if mode == "first":
        return fingers_up_single(all_hands_lm[0], hand_types[0])
    
    
    # ALL mode: one result per hand
    results = []
    for lm_list, h_type in zip(all_hands_lm, hand_types):
        results.append(fingers_up_single(lm_list, h_type))
    return results



def find_distance(lm_list, p1, p2, img=None, draw=True, r=15, t=3):
    """
    Computes Euclidean distance between two landmark points and optionally visualizes it.
    
    Args:
        lm_list (list of [id, x, y]): Output from find_positions().
        p1, p2 (int): Landmark IDs to measure between.
        img (numpy.ndarray or None): Image for drawing (optional).
        draw (bool): If True and img provided, draw line & circles.
        r (int): Radius of endpoint circles.
        t (int): Thickness of connecting line.
    
    Returns:
        tuple:
            length (float): Euclidean distance between the two points.
            info (list): [x1, y1, x2, y2, cx, cy] coordinates of measurement.
    """
    if not lm_list:
        return 0, []
    x1, y1 = lm_list[p1][1], lm_list[p1][2]
    x2, y2 = lm_list[p2][1], lm_list[p2][2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    if draw and img is not None:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
    return length, [x1, y1, x2, y2, cx, cy]


def classify_gesture(fingers):
    """
    Classifies common static hand gestures based on finger states.
    
    Args:
        fingers (list of int): Output from fingers_up().
    
    Returns:
        str: Name of detected gesture.
    """
    if not fingers:
        return "No Hand"
    patterns = {
        (0,0,0,0,0): "Fist",
        (1,1,1,1,1): "Open Hand",
        (0,1,0,0,0): "Point",
        (0,1,1,0,0): "Peace",
        (1,0,0,0,0): "Thumbs Up",
        (1,0,0,0,1): "Rock On",
        (1,1,0,0,0): "Gun"
    }
    return patterns.get(tuple(fingers), f"Custom ({sum(fingers)} up)")


def get_hand_types(results):
    """
    Returns detected hand types (Left/Right) for all hands in a frame.
    
    Args:
        results: Output from find_hands(), containing multi_handedness.
    
    Returns:
        list of str: e.g., ["Left", "Right"].
    """
    types = []
    if results.multi_handedness:
        for hand_hm in results.multi_handedness:
            original = hand_hm.classification[0].label
            types.append(original)    
    return types
