import cv2
import numpy as np
import face_recognition
import os
import time
import csv
import warnings

path = '../ImageAttendance'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodingList.append(encodes[0])
        else:
            print("Warning: No face found in image.")
    return encodingList


encodingListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
timers = {}

while True:
    success, img = cap.read()
    if not success or img is None or img.size == 0:
        print("no valid frame")
        continue

    visible_now = set()

    try:
        Small_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        Small_img = cv2.cvtColor(Small_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("Error during color conversion:", e)
        continue

    Current_face_frame = face_recognition.face_locations(Small_img)
    encodings_current_frame = face_recognition.face_encodings(Small_img, Current_face_frame)

    for encodeFace, faceLoc in zip(encodings_current_frame, Current_face_frame):
        matches = face_recognition.compare_faces(encodingListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodingListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            visible_now.add(name)

            if name not in timers:
                timers[name] = {"start_time": time.time(), "total_seconds": 0}
            elif timers[name]["start_time"] is None:
                timers[name]["start_time"] = time.time()


            current_time = int(timers[name]["total_seconds"] + (time.time() - timers[name]["start_time"]))
            hours = current_time // 3600
            minutes = (current_time % 3600) // 60
            seconds = current_time % 60
            timer_str = f"{hours:02}:{minutes:02}:{seconds:02}"


            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f"{name} {timer_str}", (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    for name in timers:
        if name not in visible_now and timers[name]["start_time"] is not None:
            elapsed = time.time() - timers[name]["start_time"]
            timers[name]["total_seconds"] += int(elapsed)
            timers[name]["start_time"] = None

    cv2.imshow("Webcam", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


#tried to implement a feature that when you stop the running the code i.e when the class finishes, it will store
# the data of every person with their time thereby checking further if they come in a certain time frame
# it is still in development
with open("face_times.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Total Time (HH:MM:SS)"])
    for name, timer_data in timers.items():
        total_seconds = timer_data["total_seconds"]
        if timer_data["start_time"] is not None:
            total_seconds += int(time.time() - timer_data["start_time"])
        hours = total_seconds / 3600
        minutes = (total_seconds % 3600) / 60
        seconds = total_seconds % 60

        writer.writerow([name, f"{hours:02}:{minutes:02}:{seconds:02}",AttendanceStatus(total_seconds)])

print("Face times saved to face_times.csv")

cap.release()
cv2.destroyAllWindows()
