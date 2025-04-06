import os, glob, cv2
import numpy as np

SRC_DIR = "dataset/Fake"
OUT_DIR = "transformed_images"

def perspective_transform(img):
    h, w = img.shape[:2]
    src_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
    dst_pts = np.float32([[0,0],[w*0.8,50],[50,h*0.9],[w*0.9,h*0.95]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w,h))
    return warped

if __name__=="__main__":
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    for f in glob.glob(os.path.join(SRC_DIR,"*.*")):
        img = cv2.imread(f)
        if img is None:
            continue
        out = perspective_transform(img)
        base = os.path.basename(f)
        cv2.imwrite(os.path.join(OUT_DIR, base), out)
    print("Applied perspective transform to images.")
