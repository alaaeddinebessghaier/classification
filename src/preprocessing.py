import cv2

def preprocess_image(img, size=(128,128)):
    """
    Redimensionne l'image et normalise les pixels entre 0 et 1
    """
    img = cv2.resize(img, size)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    return img
