'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.cap = None
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)
            
    def get_dim(self):
        try:
            self.cap.open(self.input_file)
            (width, height, fps) = (int(self.cap.get(3)), int(self.cap.get(4)), int(self.cap.get(5)))
        except:
            height, width, channels = self.cap.shape
        return (width, height, fps)
        
    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        if self.input_type=='video' or self.input_type=='cam':  
            _ = True
            while _:
                _, frame=self.cap.read()
                yield frame
        else:
            yield self.cap

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

