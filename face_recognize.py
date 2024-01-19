import cv2
import os
import numpy as np

size = 4

# Change the paths below to the location where these files are on your machine
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'C:/Users/amirb/PycharmProjects/pythonProject1/dataset'

print('Training classifier...')

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(width, height) = (130, 100)


def draw_rectangle(img, pt1, pt2, color, thickness):
    cv2.rectangle(img, pt1, pt2, color, thickness)


(images, labels) = [np.array(lists) for lists in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

print('Classifier trained!')
print('Attempting to recognize faces...')

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)

        # draw a red rectangle around the face
        draw_rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if prediction[1] < 75:
            cv2.putText(im, '%s' % (names[prediction[0]].strip()), (x + 5, (y + 25) + h),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (20, 185, 20), 2)
            confidence = (prediction[1]) if prediction[1] <= 100.0 else 100.0
            print("Predicted person: {}, Confidence: {}%".format(names[prediction[0]].strip(),
                                                                   round((confidence / 74.5) * 100, 2)))
        else:
            cv2.putText(im, 'Unknown', (x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN, 1.5, (65, 65, 255), 2)
            print("Predicted person: Unknown")

    cv2.imshow('OpenCV Face Recognition - Press esc to close', im)
    key = cv2.waitKey(10)

    if key == 27:
        break

cv2.destroyAllWindows()
