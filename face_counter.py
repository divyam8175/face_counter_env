import cv2

# Load the Haar Cascade file
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the image
image_path = r"images (1).jpg"  # Replace with the path to your image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow("Faces Detected", image)
print(f"Number of faces detected: {len(faces)}")

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
