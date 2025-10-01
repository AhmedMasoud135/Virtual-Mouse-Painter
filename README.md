# Hand Tracking Control System

A comprehensive computer vision application that enables **virtual mouse control** and **screen painting** using hand gestures. Built with OpenCV, MediaPipe, and Python, this project offers intuitive hands-free interaction with your computer.

## 🚀 Features

### **Dual Mode Operation**
- **Mouse Mode**: Control your computer's cursor using hand gestures
- **Paint Mode**: Draw directly on your screen with finger movements

### **Advanced Hand Tracking**
- Real-time hand detection and landmark tracking
- Support for both left and right hand recognition
- Accurate finger position detection and gesture classification
- Smooth cursor movement with customizable sensitivity

### **Mouse Controls**
- **Cursor Movement**: Move mouse by pointing with index finger
- **Left Click**: Pinch thumb and index finger together
- **Drag & Drop**: Hold pinch gesture between thumb and middle finger
- **Double Click**: Two quick pinch gestures
- **Scroll**: Use index and middle fingers together (with thumb down)

### **Painting Features**
- **Multi-Color Drawing**: Choose from Red, Green, Blue, Yellow, Pink colors
- **Eraser Tool**: Remove drawings with a larger brush
- **Screen Overlay**: Transparent drawing layer over your desktop
- **Real-time Drawing**: Smooth line rendering with gesture-based control

### **Gesture Recognition**
- Fist, Open Hand, Point, Peace Sign, Thumbs Up, Rock On, Gun gestures
- Dynamic gesture classification with real-time feedback
- Visual indicators for active gestures and modes

## 📋 Requirements

```
opencv-python>=4.5.0
mediapipe>=0.10.9
numpy>=1.21.0
autopy>=4.0.0
pyautogui>=0.9.50
seaborn>=0.11.0
tkinter (usually included with Python)
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hand-tracking-control-system.git
   cd hand-tracking-control-system
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe numpy autopy pyautogui seaborn
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
hand-tracking-control-system/
├── main.py                    # Main application with dual mode support
├── HandTrackingFunctions.py   # Core hand detection and tracking functions
├── MouseFunctions.py          # Mouse control implementations
├── PainterFunctions.py        # Screen painting and overlay functions
├── Mouse.py                   # Standalone mouse control application
├── MouseFunctions_Test.py     # Testing script for mouse functions
├── HandTracking_Test.py       # Testing script for hand tracking
└── README.md                  # This file
```

## 🎮 Usage

### **Getting Started**
1. Run `python main.py` to start the application
2. Position your hand in front of the camera
3. Use the selection panel at the top to switch between modes

### **Mode Selection**
- **Index + Middle Finger Up**: Access selection panel
- Click on **MOUSE** or **PAINT** buttons to switch modes
- In Paint mode, select colors from the palette

### **Mouse Control Gestures**
- **Index Finger Up**: Move cursor
- **Thumb + Index Pinch**: Left click
- **Thumb + Middle Pinch**: Drag and drop
- **Index + Middle Up**: Double click
- **Index + Middle Up (Thumb Down)**: Scroll

### **Painting Gestures**
- **Index Up + Middle Down**: Draw/Paint
- **Index + Middle Up**: Move without drawing
- Select colors and eraser from the top panel

### **Keyboard Shortcuts**
- **'q'**: Quit application
- **'c'**: Clear screen drawings (Paint mode)
- **'s'**: Save screenshot of current screen

## 🔧 Configuration

### **Camera Settings**
- Default resolution: 648x488
- Adjust `wcam` and `hcam` in configuration files

### **Mouse Sensitivity**
- Modify `SMOOTHING` factor (1-15, higher = smoother)
- Adjust `FRAME_R` for active area reduction

### **Gesture Thresholds**
- `CLICK_THRESHOLD`: Distance for click detection (default: 35)
- `DRAG_THRESHOLD`: Distance for drag detection (default: 35)
- `CLICK_HOLD_TIME`: Time to hold gesture before action (default: 0.1s)

## 🧪 Testing

Run individual test files to verify functionality:

```bash
# Test hand tracking features
python HandTracking_Test.py

# Test mouse control functions
python MouseFunctions_Test.py

# Test standalone mouse application
python Mouse.py
```

---

**⭐ Star this repository if you found it helpful!**
