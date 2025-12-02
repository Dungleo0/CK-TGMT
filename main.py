import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
from paddleocr import PaddleOCR

# paddle_ocr = PaddleOCR(lang='en')

os.makedirs("outputs", exist_ok=True)



def preprocess_barcode_for_ocr(image):
    """
    Tiền xử lý ảnh mã vạch để cải thiện độ chính xác OCR
    """
    # 1. Chuyển sang grayscale nếu cần
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 2. Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. Resize ảnh để cải thiện chất lượng
    height, width = enhanced.shape
    if height < 100:  # Nếu ảnh quá nhỏ
        scale_factor = 200 / height
        new_width = int(width * scale_factor)
        enhanced = cv2.resize(enhanced, (new_width, 200), interpolation=cv2.INTER_CUBIC)
    
    # 4. Lọc nhiễu
    denoised = cv2.medianBlur(enhanced, 3)
    
    # 5. Chuyển ngược lại thành 3 kênh cho PaddleOCR
    result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    return result

def filter_numeric_text(text):
    """
    Lọc chỉ giữ lại các ký tự số
    """
    # Loại bỏ các ký tự không phải số
    numeric_text = ''.join(filter(str.isdigit, text))
    return numeric_text


def order_points(pts):
    """Trả về 4 points theo thứ tự: tl, tr, br, bl"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def deskew_and_crop(crop, debug=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th

    # Morphological closing để gộp các vạch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    plt.figure(figsize=(6,6))
    plt.imshow(morph, cmap="gray")
    plt.title("Binary Mask")
    plt.axis("off")
    plt.show()

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop, False, (0.0, 0.0)

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = order_points(box.astype("float32"))


    w = int(rect[1][0])
    h = int(rect[1][1])
    if w == 0 or h == 0:
        return crop, False, (0.0, 0.0)

    # đảm bảo width >= height
    dst_w, dst_h = (w, h)
    if dst_w < dst_h:
        dst_w, dst_h = dst_h, dst_w

    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(crop, M, (dst_w, dst_h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # kiểm tra hướng vạch
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_w, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_w, cv2.CV_64F, 0, 1, ksize=3)
    magx = float(np.mean(np.abs(sobelx)))
    magy = float(np.mean(np.abs(sobely)))

    rotated_flag = False
    if magx < magy:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        rotated_flag = True

    if debug:
        print(f"rect (w,h,angle) = ({w},{h},{rect[2]:.2f}), magx={magx:.3f}, magy={magy:.3f}, rot90={rotated_flag}")
        # Vẽ contour box
        debug_img = crop.copy()
        box_int = box.astype(int)
        cv2.drawContours(debug_img, [box_int], -1, (0, 255, 0), 2)
        plt.figure(figsize=(8,6))
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Contour bounding box")
        plt.axis("off")
        plt.show()


    return warped, rotated_flag, (magx, magy)




# ---------- Main pipeline ----------
# 1) Load YOLO
model = YOLO("./models/barcode_detecter.pt")   # chỉnh path nếu cần
# D:\Product-Barcode-Scanner\datasets\test\images\ProductBarcode378_jpg.rf.062337523d53af94461a074010eb1d1b.jpg
# D:\Product-Barcode-Scanner\datasets\test\images\ProductBarcode526_jpg.rf.feda99061d4baa5061535a23d73ae2b0.jpg
# 2) Ảnh nguồn
# source = "./datasets/test/images/ProductBarcode526_jpg.rf.feda99061d4baa5061535a23d73ae2b0.jpg"
source = "D:\\Product-Barcode-Scanner\\test\\test_imgs\\test_1.jpg"
img = cv2.imread(source)
if img is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh: {source}")

# 3) Detect
results = model.predict(source=source, conf=0.8)
boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) and hasattr(results[0].boxes, "xyxy") else []

if len(boxes) == 0:
    print("Không phát hiện bounding box nào.")
else:

    img_detect = img.copy()
    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_detect, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_detect, f"Box {i}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite("outputs/detections.jpg", img_detect)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB))
    plt.title("YOLO Detections")
    plt.axis("off")
    plt.show()


    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box.astype(int)
        # guard against out-of-range
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        crop = img[y1:y2, x1:x2].copy()
        warped, rotated_flag, (magx, magy) = deskew_and_crop(crop, debug=True)

        # Lưu file kết quả
        cv2.imwrite(f"outputs/crop_{i}.jpg", crop)
        cv2.imwrite(f"outputs/warped_{i}.jpg", warped)

        # Hiển thị (matplotlib)
        fig, ax = plt.subplots(1, 3, figsize=(15,6))
        ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"Crop {i}")
        ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"Warped {i} (rotated_flag={rotated_flag})")
        ax[1].axis("off")

        # Nội dung debug: hiển thị map gradient trung bình
        debug_img = np.zeros_like(warped)
        ax[2].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        ax[2].set_title(f"magx={magx:.1f}, magy={magy:.1f}")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

        reader = easyocr.Reader(['en'], gpu=False)
        pre = preprocess_barcode_for_ocr(warped)
        detected_texts = reader.readtext(pre,allowlist='0123456789',threshold=0.6)
        print("EasyOCR Detected Texts (all):", detected_texts[0][1])
        # res = paddle_ocr.predict(pre)

        # print(res)
        
        # debug hiển thị ảnh tiền xử lý
        import matplotlib.pyplot as plt
        plt.imshow(pre, cmap="gray")
        plt.title("Preprocessed for OCR")
        plt.axis("off")
        plt.show()

        plt.imshow(img)
        plt.title("Code : " + detected_texts[0][1])
        plt.axis("off")
        plt.show()
