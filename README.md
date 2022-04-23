# OpenCV-yolo-stream
Real time object detection in (youtube) video stream:
- Python 3.6
- OpenCV 3.4.2

### Installation
```bash
python3 -m venv ./env
source ./env/bin/activate
pip install -r requirements.txt
```

==========================================================================================================
```bash
pip install numpy pandas argparse datetime pafy streamlink flask_opencv_streamer youtube-dl flask-opencv-streamer opencv-contrib-python
```
* OpenCV with YOLOv3 detectionmethod (https://pjreddie.com/darknet/yolo/) 
* Streamlink (https://github.com/streamlink/streamlink)

Repositori ini dapat digunakan untuk melakukan deteksi objek dalam video streaming, kemudian mencatat jumlah objek yang terdeteksi ke file output, setiap x detik (default 5 detik, tergantung pada kinerja). Ini dilakukan dengan memanfaatkan perpustakaan OpenCV dengan metode deteksi YOLOv3. Untuk pengenalan opencv dan yolo, lihat: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/. 

## YOLO Weights:
Download data YOLOv3 weights:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

Weights telah dilatih pada COCO dataset (http://cocodataset.org)

## Contoh Output Webcam Jackson Hole Town Square
 Contoh youtube streams: Jackson Hole Wyoming USA Live Cams - SeeJH.com
* https://youtu.be/1EiC9bvVGnk
* https://youtu.be/RZWzyQuFxgE


Level yang dipakai:
* confidence level 0.30
* threshold level 0.55

```
python main.py --yolo yolo-coco --url https://youtu.be/y7QiNgui5Tg
```

test