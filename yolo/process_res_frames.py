import cv2
import os

image_folder = 'results'
video_name = 'video.avi'

def format_result_frames(image_folder='results', video_name='video.avi'):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps=1.5, frameSize=(width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


