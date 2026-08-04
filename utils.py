import cv2

def load_mask(cap, mask_path):
  mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
  mask = cv2.resize(mask, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  
  return mask

def apply_mask(frame, mask):
  return cv2.bitwise_and(frame, frame, mask=mask)

def mouse_callback(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    print(f"({x}, {y})")