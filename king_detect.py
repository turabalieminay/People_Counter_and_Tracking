import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # OpenMP çakışmasını çözmek için
from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort  # Sort algoritmasını kullanacağız

# YOLOv8n modelini yükleme (Sadece insan tespiti için yolov8n.pt modelini kullanıyoruz)
model = YOLO('model/yolov8s.pt')

# Video dosyasının yolu
video_path = 'video/tester9.mp4'
output_path = 'predict.mp4'

# Video yakalama (VideoCapture) nesnesi oluşturma
cap = cv2.VideoCapture(video_path)

# Video özelliklerini alma
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video yazıcı (VideoWriter) nesnesi oluşturma
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Takip algoritması oluşturma (Sort)
tracker = Sort()

# Sort ID'lerini kendi ID'lerimize eşlemek için bir sözlük
sort_id_to_person_id = {}

# Boşta olan ID'leri tutacak liste
available_ids = []

# Şu ana kadar kullanılan en yüksek ID numarası
person_id_counter = 1  # 1'den başlayacak şekilde ayarlandı

# İnsan sınıfı ID'si (COCO veri kümesine göre insan sınıfı ID'si genellikle 0'dır)
human_class_id = 0

while cap.isOpened():
    ret, frame = cap.read()  # Her bir kareyi oku
    if not ret:
        break  # Eğer kare okunamazsa döngüden çık

    # Model ile tespit yapma
    results = model(frame)

    # Şu anki karede tespit edilen kişi kutuları ve sınıf ID'leri
    detections = []
    for box in results[0].boxes:
        if int(box.cls[0]) == human_class_id:  # Sadece insan sınıfı ID'sini tespit et
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()
            detections.append([x1, y1, x2, y2, confidence])

    # Tespit edilen kutuları numpy dizisine dönüştürme
    detections = np.array(detections)

    # Eğer tespit varsa takip algoritmasına ver
    if len(detections) > 0:
        tracked_objects = tracker.update(detections)
    else:
        tracked_objects = []

    # Mevcut kişiler listesi
    current_person_ids = []

    # Tespit edilen nesneler üzerinde numaralandırma ve çizim
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        # Sort ID'sini kendi ID'mize eşle
        if track_id not in sort_id_to_person_id:
            if available_ids:
                # Eğer boşta ID varsa, onu kullan
                person_id = available_ids.pop(0)
            else:
                # Eğer boşta ID yoksa yeni bir ID ata
                person_id = person_id_counter
                person_id_counter += 1

            sort_id_to_person_id[track_id] = person_id

        person_id = sort_id_to_person_id[track_id]
        current_person_ids.append(person_id)

        # Tespit edilen insan kutusu etrafına dikdörtgen çizme
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Kişi ID'sini kutunun üstüne yazma
        label = f"Person {person_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Ekrandan çıkmış olan kişilerin ID'lerini boşta ID listesine ekle
    for sort_id in list(sort_id_to_person_id):
        if sort_id_to_person_id[sort_id] not in current_person_ids:
            available_ids.append(sort_id_to_person_id.pop(sort_id))

    # Ekranın sol üst köşesine şu anda kadrajda bulunan insan sayısını yazdır
    current_people_count = len(current_person_ids)
    cv2.putText(frame, f"People Count: {current_people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # İşlenen kareyi video dosyasına yazma
    out.write(frame)

    # Her kareyi ekranda gösterme
    cv2.imshow('Human Detection and Counting', frame)

    # 'q' tuşuna basıldığında videoyu durdurma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma ve pencereleri kapatma
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video başarıyla işlendi ve kaydedildi: {output_path}")
