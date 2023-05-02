import cv2
import os
import glob

image_directory = "TrainReal/output-imgs"
image_files = glob.glob(os.path.join(image_directory, "*"))
image_files.sort()

# Set the delay between frames in milliseconds (e.g., 30 ms for ~33 fps)
delay = 30

for image_file in image_files:
    img = cv2.imread(image_file)
    if img is not None:
        cv2.imshow("Video", img)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    else:
        print(f"Unable to read image: {image_file}")
cv2.destroyAllWindows()