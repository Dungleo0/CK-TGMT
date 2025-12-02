from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import zxingcpp
import pandas as pd
import numpy as np


def preprocess_barcode(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def decode_barcode(crop):
    # Thử decode với ảnh gốc trước
    zxing_res = zxingcpp.read_barcodes(crop)
    if zxing_res:
        return zxing_res[0].text
    
    # Nếu không thành công, thử với ảnh đã tiền xử lý
    try:
        processed_crop = preprocess_barcode(crop)
        zxing_res = zxingcpp.read_barcodes(processed_crop)
        if zxing_res:
            return zxing_res[0].text
    except Exception as e:
        print(f"Error in preprocessing: {e}")
    
    return None


def check_barcode(product_dataset, barcode):
    result = product_dataset[product_dataset["Barcode"].astype(str).str.strip() == str(barcode).strip()]
    if not result.empty:
        return result.iloc[0]
    return None


def main():
    # Load model + DeepSORT
    model = YOLO("models/barcode_detecter.pt")
    tracker = DeepSort(max_age=5, n_init=2, max_iou_distance=0.9)
    product_dataset = pd.read_csv("App/product_data/product.csv")
    video_path = "test/test_videos/test_vd1.mp4"

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Can't open")
        return

    locked_barcodes = {}  # track_id -> {"barcode": str, "bbox": (x1,y1,x2,y2)}

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # === YOLO detect ===
        results = model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        detections = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box[:4])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "barcode"))

        # === DeepSORT tracking ===
        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = t.track_id
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            barcode_text = None
            color = (0, 255, 255)

            if track_id in locked_barcodes:
                barcode_text = locked_barcodes[track_id]["barcode"]
                color = (0, 0, 255)
                
                product_info = check_barcode(product_dataset, barcode_text)
                if product_info is not None:
                    cv2.putText(frame, f"Barcode: {barcode_text}", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame,f"conf: {t.det_conf:.2f}", (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"{product_info['Name']}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"{product_info['Weight']} kg", (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"{product_info['Price']} dong", (x1, y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(frame, "Unknown", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    barcode_text = decode_barcode(crop)

                if barcode_text:
                    locked_barcodes[track_id] = {"barcode": barcode_text, "bbox": (x1, y1, x2, y2)}
                    color = (0, 255, 0)
                    print(f"Locked track {track_id} with barcode {barcode_text}")
                else:
                    cv2.putText(frame, "Cannot scan this barcode", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            h, w = frame.shape[:2]
            if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
                if track_id in locked_barcodes:
                    print(f" Unlock track {track_id} - barcode {locked_barcodes[track_id]['barcode']}")
                    del locked_barcodes[track_id]

        frame = cv2.resize(frame, (700, 750))
        cv2.imshow("CV ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()