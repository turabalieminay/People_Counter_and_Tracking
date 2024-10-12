from ultralytics import YOLO
import cv2
import os

# Modeli yükleme
model = YOLO('model/people_detect.pt')

# Girdi ve çıktı klasörleri
input_folder = 'C:/Users/aytur/Desktop/people_Detect/inference'
output_folder = 'C:/Users/aytur/Desktop/people_Detect/predict'

# Çıktı klasörü yoksa oluşturma
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Girdi klasöründeki tüm resim dosyalarını işleme
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".webp") or filename.endswith(".jfif"):
        # Resmin tam yolunu oluşturma
        image_path = os.path.join(input_folder, filename)
        
        # Resmi yükleme
        image = cv2.imread(image_path)
        
        # Model ile tespit yapma
        results = model(image)

        # Tespit edilen kutu koordinatları, güven skoru ve sınıf bilgisi
        detections = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Tespit edilen nesneler üzerinde numaralandırma ve çizim
        for idx, (box, conf, cls) in enumerate(zip(detections, confidences, classes)):
            x1, y1, x2, y2 = [int(coord) for coord in box]  # Koordinatları tam sayıya yuvarlama
            
            # Tespit edilen insan kutusu etrafına dikdörtgen çizme
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Numarayı ve güven skorunu kutunun üstüne daha küçük punto ile yazma
            label = f"{idx + 1} Conf: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Çıktı yolunu oluşturma
        output_path = os.path.join(output_folder, filename)

        # İşlenmiş resmi çıktıya kaydetme
        cv2.imwrite(output_path, image)
        print(f"{filename} kaydedildi: {output_path}")

print("Tüm resimler başarıyla işlendi ve kaydedildi.")
