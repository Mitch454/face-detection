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

    # Load the PNG image you want to overlay (change the filename as needed)
    overlay_image = cv2.imread("overlay.png", cv2.IMREAD_UNCHANGED)

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

            for (x, y, w, h) in faces:
                # Calculate the aspect ratio of the overlay image
                aspect_ratio = overlay_image.shape[1] / overlay_image.shape[0]

                # Calculate the new width and height of the overlay image while maintaining the aspect ratio
                new_width = int(h * aspect_ratio)
                new_height = h

                # Ensure that the resized overlay image fits within the face region
                if new_width > w:
                    new_width = w
                    new_height = int(w / aspect_ratio)

                # Resize the overlay image
                overlay_resized = cv2.resize(overlay_image, (new_width, new_height))

                # Calculate the position to place the overlay image in the center of the face region
                x_offset = x + (w - new_width) // 2
                y_offset = y + (h - new_height) // 2

                # Overlay the resized image on the screenshot using the mask
                result = screenshot.copy()
                for c in range(0, 3):
                    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] = (
                        screenshot[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] *
                        (1 - overlay_resized[:, :, 3] / 255.0) +
                        overlay_resized[:, :, c] * (overlay_resized[:, :, 3] / 255.0)
                    )

                screenshot = result

            # Display the output image
            cv2.imshow("Face Detection", screenshot)

            # Wait for the 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cv2.destroyAllWindows()

# Run the face detection function
detect_face()
