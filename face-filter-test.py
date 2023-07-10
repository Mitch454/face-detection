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

    # Load the filter image
    filter_image = cv2.imread("filter.png", -1)
    filter_image_width = 200
    filter_image_height = 200

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
            
            # Overlay the filter on the detected faces
            for (x, y, w, h) in faces:
                # Resize the filter image to match the size of the detected face
                resized_filter = cv2.resize(filter_image, (w, h))
                
                # Define the region of interest (ROI) for the filter
                roi = screenshot[y:y+h, x:x+w]
                
                # Create a mask from the alpha channel of the filter image
                mask = resized_filter[:, :, 3] / 255.0
                
                # Reverse the mask for inverse filtering
                inverse_mask = 1.0 - mask
                
                # Apply the filter to the ROI
                filtered_roi = cv2.multiply(roi.astype(float), inverse_mask)
                filtered_filter = cv2.multiply(resized_filter[:, :, 0:3].astype(float), mask[:, :, np.newaxis])
                
                # Add the filtered ROI and filter together
                result = cv2.add(filtered_roi, filtered_filter)
                
                # Convert the result back to the original data type and assign it to the ROI
                screenshot[y:y+h, x:x+w] = result.astype(np.uint8)
            
            # Display the output image
            cv2.imshow("Face Detection", screenshot)
            
            # Wait for the 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cv2.destroyAllWindows()

# Run the face detection function
detect_face()