from ultralytics import YOLO
import cv2

# Modeli yükleme
model = YOLO('model/people_detect.pt')

# Resmi yükleme
image_path = 'inference/Ekran görüntüsü 2024-09-29 014738.png'
image = cv2.imread(image_path)

# Model ile tespit yapma
results = model(image)

# Tespit edilen kutu koordinatları, güven skoru ve sınıf bilgisi
detections = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] şeklinde
confidences = results[0].boxes.conf.cpu().numpy()  # Güven skoru
classes = results[0].boxes.cls.cpu().numpy()  # Sınıf bilgisi

# Tespit edilen nesneler üzerinde numaralandırma ve çizim
for idx, (box, conf, cls) in enumerate(zip(detections, confidences, classes)):
    x1, y1, x2, y2 = [int(coord) for coord in box]  # Koordinatları tam sayıya yuvarlama
    
    # Tespit edilen insan kutusu etrafına dikdörtgen çizme
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Numarayı ve güven skorunu kutunun üstüne yazma
    label = f"{idx + 1} Conf: {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Tespit edilen sonuçları gösterme
cv2.imshow("Detection Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sonuçları kaydetme
output_path = 'people_detect_result.jpg'
cv2.imwrite(output_path, image)

print(f"Sonuç kaydedildi: {output_path}")
