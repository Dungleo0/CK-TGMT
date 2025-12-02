from matplotlib.pyplot import gray
from ultralytics import YOLO
import cv2
import zxingcpp
import numpy as np
import pandas as pd


def preprocess_barcode(crop):
    """Tăng cường độ tương phản và làm sắc nét barcode trước khi decode."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Tăng tương phản cục bộ
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Phóng to để dễ đọc hơn
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

    # Làm sắc nét
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    return sharp

def detect_inverted_by_gradient(gray):
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # ngang
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)  # dọc

    top_score = np.mean(np.abs(sobel_y[:h//4, :]))
    bottom_score = np.mean(np.abs(sobel_y[3*h//4:, :]))

    if bottom_score < top_score:
        return True  # bị ngược
    return False


def check_barcode(product_dataset, barcode):
    """Kiểm tra mã barcode có tồn tại trong file CSV sản phẩm không."""
    try:
        barcode_int = int(barcode)
    except ValueError:
        return None

    result = product_dataset[product_dataset["Barcode"] == barcode_int]
    return result.iloc[0] if not result.empty else None


def main():
    # --- 1. Load mô hình YOLO ---
    model = YOLO("models/barcode_detecter.pt")

    # --- 2. Load ảnh và dữ liệu sản phẩm ---
    img = cv2.imread("test/test_imgs/test_3.jpg")
    product_dataset = pd.read_csv("App/product_data/product.csv")

    # --- 3. Phát hiện mã vạch ---
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    
    
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img[y1:y2, x1:x2]
        # crop = preprocess_barcode(crop)

        # h, w = crop.shape[:2]

        # if h > w:
        #     print("xoay 90")
        #     crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        


        cv2.imshow(f"Cropped Barcode {conf}", crop)
        cv2.waitKey(1)

        if detect_inverted_by_gradient(crop):
            print("xoay 180")
            crop_new = cv2.rotate(crop, cv2.ROTATE_180)

            cv2.imshow(f"rotate Barcode {conf:.2f}", crop_new)
            cv2.waitKey(1)

        # --- 4. Giải mã barcode ---
        decoded = zxingcpp.read_barcodes(crop)
        barcode_text = None

        if decoded:
            for res in decoded:
                barcode_text = res.text
                print(f"Code: {res.text} | Type: {res.format}")
        else:
            print(f"Non ({x1},{y1},{x2},{y2})")

        # --- 5. Vẽ bounding box và hiển thị thông tin ---
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Code: {barcode_text}", (x1, y2+100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if barcode_text:
            product_info = check_barcode(product_dataset, barcode_text)
            
           
            
            if product_info is not None:
                name = product_info["Name"]
                weight = product_info["Weight"]
                price = product_info["Price"]
                

                cv2.putText(img, f"{name}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"{weight} kg", (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"{price} dong", (x1, y2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                pass
                # cv2.putText(img, f"{barcode_text}", (x1, y2 + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(img, "Unknown", (x1, y2 + 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- 6. Hiển thị kết quả ---
    img_re = cv2.resize(img, (800, 600))
    cv2.imshow("Detected", img_re)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
