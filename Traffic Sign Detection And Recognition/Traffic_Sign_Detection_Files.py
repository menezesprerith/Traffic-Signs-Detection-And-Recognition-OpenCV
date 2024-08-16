import cv2 as cv
import os

def main(image_paths):
    stop_sign_haar_cascade = cv.CascadeClassifier('stopsign_classifier_haar.xml')
    bump_sign_haar_cascade = cv.CascadeClassifier('bumpersign_classifier_haar.xml')
    speed_limit_haar_cascade = cv.CascadeClassifier('speed_limit_haar_cascade.xml')
    walkway_haar_cascade = cv.CascadeClassifier('walkway_haar_cascade.xml')
    traffic_light_haar_cascade = cv.CascadeClassifier('traffic_light_haar_cascade.xml')
    yield_haar_cascade = cv.CascadeClassifier('yield_haar_cascade.xml')
    leftTurn_haar_cascade = cv.CascadeClassifier('turnLeft_ahead_haar_classifier.xml')
    rightTurn_haar_cascade = cv.CascadeClassifier('turnRight_ahead_haar_classifier.xml')


    try:
        for image_path in image_paths:
            img = cv.imread(image_path)
            if img is None:
                print(f"Failed to load image at {image_path}")
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            stop_sign_rects = stop_sign_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
            bump_sign_rects = bump_sign_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
            speed_limit_rects = speed_limit_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
            walkway_rects = walkway_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
            traffic_light_rects = traffic_light_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
            yield_rects = yield_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
            leftTurn_rects = leftTurn_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
            rightTurn_rects = rightTurn_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

            for (x, y, w, h) in stop_sign_rects:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                cv.putText(img, 'Stop Sign', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            for (x, y, w, h) in bump_sign_rects:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                cv.putText(img, 'Road Hump Ahead', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            for (x, y, w, h) in speed_limit_rects:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
                cv.putText(img, 'Speed Limit', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            for (x, y, w, h) in walkway_rects:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), thickness=2)
                cv.putText(img, 'Walk Way', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            for (x, y, w, h) in traffic_light_rects:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), thickness=2)
                cv.putText(img, 'Traffic Light', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            for (x, y, w, h) in yield_rects:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), thickness=2)
                cv.putText(img, 'Give Way Sign', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            for (x, y, w, h) in rightTurn_rects:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), thickness=2)
                cv.putText(img, 'Right Turn Ahead', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            for (x, y, w, h) in leftTurn_rects:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), thickness=2)
                cv.putText(img, 'Left Turn Ahead', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)



            cv.imshow('Detected', img)
            cv.waitKey(0)  # Press any key to move to the next image

    except Exception as e:
        print(f"An error occurred: {e}")

    cv.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to your images here
    image_folder = '/path/to/your/images'
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    main(image_files)
