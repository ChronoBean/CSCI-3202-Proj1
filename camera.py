import cv2
import model

def main():
    # Initialize camera capture
    cap = cv2.VideoCapture(0)  # Adjust the index if multiple cameras are present

    # Load AI model
    emotion_model = model.load_model()  # Implement the `load_model` function in `model.py`

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform facial expression classification
        expression = model.classify_expression(frame, emotion_model)  # Implement the `classify_expression` function in `model.py`

        # Draw expression text on the frame
        cv2.putText(frame, expression, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Facial Expression Recognition', frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()