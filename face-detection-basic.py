import cv2
import numpy as np
import mss
import mss.tools
import time


def detect_face():
    # Set the monitor coordinates for the region of interest (ROI)
    monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

    # Load the pre-trained face detection cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Create an instance of MSS (Multiple ScreenShot) to capture frames
    with mss.mss() as sct:
        # Create a named window to display the output
        cv2.namedWindow("Face Detection")
        
        while True:
            # Capture a screenshot of the specified region of the screen
            screenshot = np.array(sct.grab(monitor))
            
            # Convert the screenshot to grayscale for face detection
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Perform face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(screenshot, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the output image
            cv2.imshow("Face Detection", screenshot)
            
            # Wait for the 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cv2.destroyAllWindows()

# Run the face detection function
detect_face()