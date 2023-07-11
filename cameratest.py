import cv2

def open_camera():
    cap = cv2.VideoCapture(0)  # 0 indicates the default camera

    while True:
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Camera', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    open_camera()
