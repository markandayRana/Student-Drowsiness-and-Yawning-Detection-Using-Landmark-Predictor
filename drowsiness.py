from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import serial
import RPi.GPIO as GPIO
import smbus
import urllib.request

# GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
m1 = 19
ac =2
buz =26
GPIO.setup(buz, GPIO.OUT)
GPIO.setup(m1,GPIO.OUT)
GPIO.setup(ac, GPIO.IN)
GPIO.output(buz,0)
kk=0
def GPS_Info():
    global NMEA_buff
    global lat_in_degrees
    global long_in_degrees
    nmea_time = []
    nmea_latitude = []
    nmea_longitude = []
    nmea_time = NMEA_buff[0]                    #extract time from GPGGA string
    nmea_latitude = NMEA_buff[1]                #extract latitude from GPGGA string
    nmea_longitude = NMEA_buff[3]               #extract longitude from GPGGA string
    
    #print("NMEA Time: ", nmea_time,'\n')
    #print ("NMEA Latitude:", nmea_latitude,"NMEA Longitude:", nmea_longitude,'\n')
    try:
        lat = float(nmea_latitude)                  #convert string into float for calculation
        longi = float(nmea_longitude)               #convertr string into float for calculation
    except:
        lat=0
        longi=0
    lat_in_degrees = convert_to_degrees(lat)    #get latitude in degree decimal format
    long_in_degrees = convert_to_degrees(longi) #get longitude in degree decimal format

def convert_to_degrees(raw_value):
    decimal_value = raw_value/100.00
    degrees = int(decimal_value)
    mm_mmmm = (decimal_value - int(decimal_value))/0.6
    position = degrees + mm_mmmm
    position = "%.4f" %(position)
    return position


gpgga_info = "$GPGGA,"
ser = serial.Serial ("/dev/ttyS0")              #Open port with baud rate
GPGGA_buffer = 0
NMEA_buff = 0
lat_in_degrees = 0
long_in_degrees = 0

##def alarm(msg):
##    global alarm_status
##    global alarm_status2
##    global saying
##
##    while alarm_status:
##        print('call')
##        s = 'espeak "' + msg + '"'
##        os.system(s)
##
##    if alarm_status2:
##        print('call')
##        saying = True
##        s = 'espeak "' + msg + '"'
##        os.system(s)
##        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Argument parser setup
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 1
YAWN_THRESH = 10
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
# vs= VideoStream(usePiCamera=True).start()       #For Raspberry Pi
time.sleep(1.0)
GPIO.output(m1,1)
ii = 0

while True:
    received_data = (str)(ser.readline())                   #read NMEA string received
    GPGGA_data_available = received_data.find(gpgga_info) 
    if(kk==0):
        lat_in_degrees=0
        lat_in_degrees=0
    if (GPGGA_data_available>0):
        kk=1
        GPGGA_buffer = received_data.split("$GPGGA,",1)[1]  #store data coming after "$GPGGA," string 
        NMEA_buff = (GPGGA_buffer.split(','))               #store comma separated data in buffer
        GPS_Info()                                          #get time, latitude, longitude
        map_link = 'http://maps.google.com/?q=' + str(lat_in_degrees) + ',' + str(long_in_degrees)    #create link to plot location on Google map
            
    map_link = 'http://maps.google.com/?q=' + str(16.5085862) + ',' + str(80.6531247)    #create link to plot location on Google map
   # print(map_link)
    print()
    aval=1-GPIO.input(ac)
    print("ACCIDENT:"+ str(aval))
    time.sleep(0.2)
    if(aval==1):
        print('Accident occured sending info')
        GPIO.output(m1,0)
        GPIO.output(buz,1)
        urllib.request.urlopen("https://api.thingspeak.com/update?api_key=43J0P0NIGILXMDDA&field1=1&field2=16.5085862&field3=80.6531247")
    frame = vs.read()

    if frame is None:
        print("Error: Frame not captured correctly.")
        continue

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                #if not alarm_status:
                   # alarm_status = True
                   # t = Thread(target=alarm, args=('Alert and wake up sir',))
                    GPIO.output(buz, 1)
##                    t.daemon = True
##                    t.start()
                    ii += 1
                    if ii > 4:
                        print('drowsiness detected sending info')
                        GPIO.output(m1,0)
                        GPIO.output(buz,1)
                        urllib.request.urlopen("https://api.thingspeak.com/update?api_key=43J0P0NIGILXMDDA&field1=2&field2=16.5085862&field3=80.6531247")

                        while True:
                            time.sleep(1)

            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

##        if distance > YAWN_THRESH:
##            cv2.putText(frame, "Yawn Alert", (10, 30),
##                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
##            if not alarm_status2 and not saying:
##                alarm_status2 = True
##                t = Thread(target=alarm, args=('Alert and take some fresh air sir',))
##                t.daemon = True
##                t.start()
##                ii += 1
##                if ii > 2:
##                    GPIO.output(m1, 0)
##                    while True:
##                        time.sleep(1)
##        else:
##            alarm_status2 = False
##
##        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
##                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
##        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
##                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
