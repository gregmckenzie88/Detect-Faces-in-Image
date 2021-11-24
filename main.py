import cv2

# create image object
image = cv2.imread('humans.jpeg', 1)

# create our cascade -- in this case
# the faces.xml will detect patterns in the image
face_cascade = cv2.CascadeClassifier('faces.xml')

# we call detectMultiScale with following parameters:
# image object containing faces
# the scale for each check, since we want to detect big and small faces in the image
# number of pixels surrounding each face to determine border
faces = face_cascade.detectMultiScale(image, 1.1, 4)

print(faces)

# each face contains x & y coordinates as well as width
# and height of each border box around the face
for (x, y, w, h) in faces:
  # draw the rectangle for each face
  # image obj, coordinate, with & height, color of rect, width of rect
  cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 4)

cv2.imwrite('human_faces.jpeg', image)
