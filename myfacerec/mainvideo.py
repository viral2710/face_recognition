import face_recognition
import os
import cv2
KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.5
image_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
video=cv2.VideoCapture("zoom_2.mp4")
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('outputfxx.avi', fourcc, 25, (1280, 720))
# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color
print('Loading known faces...')
known_faces = []
known_names = []
# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):
    # Next we load every file of faces of known person# Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only
        # (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]
        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)
next_id=0
frame_number = 0
print('Processing video...')
while True:
    # Grab a single frame of video
    ret, frame = video.read()
    frame_number += 1
     # Quit when the input video file ends
    if not ret:
        break
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    next_id=0
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        tol=face_recognition.api.face_distance(known_faces, face_encoding)
        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        for i in range(tol.size):
            tol[i]=1-tol[i]
        name = None
        if True in match:
            name=known_names[match.index(True)]+"\t"+str(tol[match.index(True)])
        face_names.append(name)
    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)
video.release()
cv2.destroyAllWindows()
