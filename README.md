Hand Tracking Hearts
Python, OpenCV ve MediaPipe kullanarak kameradan el takibi yapan küçük bir görüntü işleme projesi.
Sol elin avuç içi kameraya gösterildiğinde ekranda renkli kalpler çıkar. El iskeleti, parmak noktaları ve işaret parmağı koordinatı canlı kamera görüntüsünün üzerine çizilir.
Özellikler
Kameradan gerçek zamanlı görüntü alma
MediaPipe ile el landmark noktalarını takip etme
Sağ ve sol el etiketlerini gösterme
El iskeleti ve el kutusu çizme
İşaret parmağı ucunun koordinatını gösterme
Sol açık avuç algılandığında büyük renkli kalpler çıkarma
FPS göstergesi
Kullanılan Teknolojiler
Python
OpenCV
MediaPipe
NumPy
Kurulum
Önce projeyi bilgisayarına indir veya GitHub reposunu klonla.
Ardından proje klasöründe terminali aç ve gerekli paketleri kur:
py -m pip install -r requirements_hand_tracking.txt
Sanal ortam kullanmak istersen:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements_hand_tracking.txt
Çalıştırma
py hand_tracking.py --show-fps
Kamera açılmazsa farklı kamera numarası deneyebilirsin:
py hand_tracking.py --camera 1 --show-fps
Kalplerin hangi elde çıkacağını değiştirmek için:
py hand_tracking.py --heart-hand left --show-fps
py hand_tracking.py --heart-hand right --show-fps
py hand_tracking.py --heart-hand both --show-fps
Programı kapatmak için kamera penceresindeyken q veya ESC tuşuna bas.
Dosyalar
hand_tracking.py
requirements_hand_tracking.txt
README.md
hand_landmarker.task dosyasını GitHub'a yüklemek zorunda değilsin. Program ilk çalıştırmada bu modeli otomatik indirmeye çalışır.
GitHub'a Yüklememen Önerilen Dosyalar
.venv/
__pycache__/
hand_landmarker.task
İstersen bunları engellemek için .gitignore dosyası oluşturabilirsin:
.venv/
__pycache__/
*.pyc
hand_landmarker.task
Not
Bu proje öğrenme amacıyla hazırlanmıştır. Kamera izni, ışık miktarı ve elin kameraya olan açısı algılamayı etkileyebilir.
