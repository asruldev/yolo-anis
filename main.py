# Jalankan pada terminal (tanpa output files)
# python main.py --yolo yolo-coco --url https://youtu.be/1EiC9bvVGnk
# python main.py --yolo yolo-coco --url https://youtu.be/f8WKi3Rz3Yw
# python main.py --yolo yolo-coco --url https://youtu.be/VQKIK3XgxGs
# python main.py --yolo yolo-coco --url https://youtu.be/XoNZZJyRpUc
# python main.py --yolo yolo-coco --url https://youtu.be/y7QiNgui5Tg


# Jalankan pada terminal (dengan output file)
# python main.py --yolo yolo-coco --url https://youtu.be/1EiC9bvVGnk --output output/ouput_videosteam.avi --data output/CSV/data_videosteam.csv 


# import package yang diperlukan
import numpy as np
import pandas as pd
import argparse
import time
import datetime
import cv2
import os
import pafy
import streamlink
from flask_opencv_streamer.streamer import Streamer

# port service web
port = 4455
require_login = False
streamer = Streamer(port, require_login)

# setting argumen untuk terminal
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", required=True,
    help="video url")
ap.add_argument("-p", "--period", type=float, default=5,
    help="execution period")
ap.add_argument("-o", "--output", required=False,
    help="path to output video")
ap.add_argument("-d", "--data", required=False,
    help="path to output csv")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to yolov weights, cfg and coco directory")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.55,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# setting priode eksekusi
period = args["period"]

# muat label kelas COCO model YOLO yang telah dilatih
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# menginisialisasi daftar warna untuk mewakili setiap label kelas yang mungkin
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# dapatkan jalur ke bobot YOLO dan konfigurasi model
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# memuat detektor objek YOLO kami yang dilatih pada dataset COCO (80 kelas) 
# dan hanya menentukan nama lapisan *output* yang kami butuhkan dari YOLO
print("Initializing...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

url = args["url"]

vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")
streams = streamlink.streams(url)

# setting parameter inisialisasi video
writer = None
(W, H) = (None, None)
starttime=time.time()
frame_ind = 0
obj = np.zeros((1000,7))
# looping semua frame dari video
while True:
    # baca frame berikutnya dari file
    framedatetime = datetime.datetime.now()
    framedatetime = framedatetime.strftime('%Y%m%d%H%M%S')
    cap = cv2.VideoCapture(streams["best"].url)
    (grabbed,frame) = cap.read()
    #(grabbed, frame) = vs.read()

    # jika frame tidak terambil, maka program telah mencapai akhir video
    if not grabbed:
        break

    # jika dimensi frame kosong, ambil dari frame
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    # membuat blob dari frame yang dimuat dan lakukan forward pass dari deteksi objek YOLO, 
    # memberikan kotak dan probabilitas terkait 
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # menginisialisasi daftar kotak batas yang terdeteksi, confidences, dan class Id masing-masing
    boxes = []
    confidences = []
    classIDs = []

    # looping setiap output layer
    for output in layerOutputs:
        # looping setiap deteksi
        for detection in output:
            # ekstrak class ID dan confidence (yaitu, probability) dari deteksi objek sekarang
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # menyaring prediksi yang lemah dengan memastikan probabilitas yang terdeteksi 
            # lebih besar dari probabilitas minimum
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    #set initial objects to 0
    cars = 0
    trucks = 0
    busses = 0
    # ensure at least one detection exists
    if len(idxs) > 0:

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # check for specific objects
            if ("{}".format(LABELS[classIDs[i]]) == "car") or ("{}".format(LABELS[classIDs[i]]) == "truck") or ("{}".format(LABELS[classIDs[i]]) == "bus"):
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                # count specific objects
                if "{}".format(LABELS[classIDs[i]]) == "car":
                  cars+=1
                if "{}".format(LABELS[classIDs[i]]) == "truck":
                  trucks+=1
                if "{}".format(LABELS[classIDs[i]]) == "bus":
                  busses+=1
    # construct a tuple of information we will be displaying on the frame
    info = [
        ("Busses", busses),
        ("Trucks", trucks),
        ("Cars", cars),
    ]
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 30) + 30)),
            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    # check if video output directory is given
    if args["output"] is not None:
        # check if the video writer is None
        if writer is None:
          # initialize our video writer
          fourcc = cv2.VideoWriter_fourcc(*"MJPG")
          writer = cv2.VideoWriter(args["output"], fourcc, 30,
              (frame.shape[1], frame.shape[0]), True)
        #write the output frame to disk
        writer.write(frame)

    # check if data output directory is given
    if args["data"] is not None:
        # write number of detections to array
        obj[frame_ind][0] = int(frame_ind+1)
        obj[frame_ind][1] = len(idxs)
        obj[frame_ind][2] = int(cars)
        obj[frame_ind][3] = int(trucks)
        obj[frame_ind][4] = int(busses)
        obj[frame_ind][5] = int(framedatetime)
        # save obj as csv every 10 frames
        if frame_ind % 10 == 0:
          obj_df = pd.DataFrame(obj)
          obj_df.columns = ['Frame', 'Objects', 'Cars', 'Trucks', 'Busses', 'DateTime']
          obj_df.to_csv(args["data"])

    # print object detection info 
    print("frame: {:.0f}".format(int(frame_ind+1)), "   datetime:", str(framedatetime))
    print("                cars: {:.0f}".format(int(cars)))
    print("              trucks: {:.0f}".format(int(trucks)))
    print("              busses: {:.0f}".format(int(busses)))
    

    # wait, if period is not over jet
    time.sleep(period - ((time.time() - starttime) % period))
    # show the output frame
    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    frameR = cv2.resize(frame, (960, 540))
    #cv2.imshow("Frame", frameR)
    
    #stream to port
    streamer.update_frame(frameR)
    if not streamer.is_streaming:
        streamer.start_streaming()
    cv2.waitKey(30)

    frame_ind += 1

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# check if data output directory is given
if args["data"] is not None:
    #save obj as csv 
    obj_df = pd.DataFrame(obj)
    obj_df.columns = ['Frame', 'Objects', 'Cars', 'Trucks', 'Busses', 'DateTime']
    obj_df.to_csv(args["data"]) 

cv2.destroyAllWindows()


