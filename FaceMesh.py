import cv2
import mediapipe as mp


# noinspection PyUnresolvedReferences
def main():
    # Initialize MediaPipe FaceMesh and VideoCapture
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10
                                                   ,
                                                   min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or provide the path to a video file

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB and process it with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh
                for idx, landmark in enumerate(face_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 145), -1)

        cv2.imshow('Face Mesh Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
