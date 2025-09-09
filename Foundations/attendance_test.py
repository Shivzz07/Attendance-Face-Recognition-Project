import cv2
import numpy as np
import face_recognition
import os

# === Step 1: Load Known Faces ===
path = '../ImageAttendance'
images = []
classNames = []

mylist = os.listdir(path)
print("Found images:", mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"⚠️ Couldn't read image: {cl}")

print("Class Names:", classNames)

def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Avoid crashing on images with no faces
            encodingList.append(encode[0])
        else:
            print("⚠️ No face found in an image, skipping.")
    return encodingList

encodingListKnown = findEncodings(images)
print('✅ Encoding Complete')

# === Step 2: Setup Webcam ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW avoids MSMF glitches
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# === Step 3: Real-time Face Recognition ===
while True:
    success, img = cap.read()
    if not success or img is None:
        print("❌ Failed to grab frame from webcam.")
        break

    # Resize and convert for faster processing
    Small_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    Small_img = cv2.cvtColor(Small_img, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    Current_face_frame = face_recognition.face_locations(Small_img)
    encodings_current_frame = face_recognition.face_encodings(Small_img, Current_face_frame)

    for encodeFace, faceLoc in zip(encodings_current_frame, Current_face_frame):
        matches = face_recognition.compare_faces(encodingListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodingListKnown, encodeFace)
        print("Face distances:", faceDis)

        # Identify best match
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print("✅ Recognized:", name)

            # Draw rectangle & label
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Show webcam feed
    cv2.imshow("Webcam", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
