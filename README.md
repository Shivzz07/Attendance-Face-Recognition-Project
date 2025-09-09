Attendance Face Recognition Project

This project is an AI-powered attendance system that uses face recognition to automatically identify people through a webcam. It doesn’t just mark who is present — it also tracks how long each person stays in real time. At the end of the session, it saves everything neatly into a CSV file for record-keeping.

🚀 Features

1.Real-time face detection & recognition using OpenCV and face_recognition.
2.Tracks attendance + duration (how long someone was present).
3.Shows live name and timer on the video feed.
4.Automatically generates a CSV file with attendance logs.
5.Can recognize and track multiple people at once.

🛠️ Tech Stack

1.Python 3
2.OpenCV – for video capture and image processing
3.face_recognition – for detecting and encoding faces
4.NumPy – numerical operations
5.CSV – for saving attendance data

📂 Project Structure
Attendance-Face-Recognition-Project/
│
├── ImageAttendance/        # Folder with reference images (one per person)
├── face_recognition.py     # Main script (webcam + recognition + timer)
├── face_times.csv          # Auto-generated CSV report after each session
└── README.md               # Project documentation

⚙️ How It Works
-> Add reference images of each person inside the ImageAttendance folder.
-> Filename = Person’s name (e.g., shivam.jpg → "SHIVAM").
-> Run the script:
-> python face_recognition.py

The webcam will open, detect faces, and show each person’s name + timer.
Press q to stop the program.
A file called face_times.csv will be created with everyone’s attendance duration.

📊 Example Output (CSV) (In development stage)
Name	Total Time (HH:MM:SS)
SHIVAM	01:25:32
THE_ROCK	00:53:10

🎯 Use Cases
~ Classroom attendance tracking
~ Office meeting logs
~ Workshop & event participation tracking

✅ Future Improvements
Add an attendance status (Present / Late / Absent).
Integrate with Google Sheets or an LMS.
Add a GUI dashboard for teachers/admins.
