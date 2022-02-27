import os
import cv2


# take a video and decompose it frame-wise saving
# every frame in the given (created) folder
def decompose_video_to_frames(video_name):
    # make the folder name the same with the video_name + _frames
    folder_name = video_name.split('.')[0] + '_frames'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    else:
        print('> folder ({}) already exists'.format(folder_name))
        return folder_name

    vidcap = cv2.VideoCapture(video_name)    
    success, image = vidcap.read()
    
    count = 0
    while success:
        cv2.imwrite(folder_name + '/frame{:04d}.jpg'.format(count), image)
        success, image = vidcap.read()
        print('.', end=' {}\n'.format(count // 100) if count > 0 and count % 100 == 0 else '')
        count += 1
    print('\nframes in total =', count)

    return folder_name


if __name__ == '__main__':
    pass
    # decompose_video_to_frames('dashcam.mp4')
