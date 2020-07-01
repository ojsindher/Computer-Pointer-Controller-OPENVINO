import os
import cv2
import time
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import Model_fd
from gaze_estimation import Model_ge
from facial_landmarks_detection import Model_fld
from head_pose_estimation import Model_hpe
from mouse_controller import MouseController
from argparse import ArgumentParser

def build_parser():
    parser = ArgumentParser()
    required = parser.add_argument_group('required', 'These are must provide arguments for the main.py script')
    optional = parser.add_argument_group('optional', 'These are optional arguments as there is default values set in the app itself')
    
    optional.add_argument("-d", "--device", type=str, default="CPU", help="Specify the target device to infer on: CPU, GPU, FPGA or                                               MYRIAD  is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)")
    optional.add_argument("-c", "--prob_threshold", type=float, default=0.5, help="This specifies the probability threshold value for                                           face detection model")
    optional.add_argument("-FDO", type=int, default=0, help="to toggle displaying face detector bounding boxes")
    optional.add_argument("-FLD", type=int, default=0, help="to toggle displaying eyes bounding boxes")
    
    required.add_argument("-t", "--input_type", required=True, type=str, help="This specifies the type of input whether it can be an image,                                       or pre-saved videos, or the feed from a webcam")
    required.add_argument("-f", "--model_fd", required=True, type=str, help="Path to model's directory with a trained model for face                                             detection.")
    required.add_argument("-g", "--model_ge", required=True, type=str, help="Path to to model's directory with a trained model for gaze                                           estimation.")
    required.add_argument("-p", "--model_hpe", required=True, type=str, help="Path to to model's directory with a trained model for head                                         pose estimation.")
    required.add_argument("-l", "--model_fld", required=True, type=str, help="Path to to model's directory with a trained model for facial                                       landmarks detection.")
    required.add_argument("-i", "--input", required=True, type=str, help="Path to image or video file")
    
    
    return parser
    
def main(args):
    device = args.device
    input_type = args.input_type
    input_file = args.input
    model_fd = args.model_fd
    model_ge = args.model_ge
    model_hpe = args.model_hpe
    model_fld = args.model_fld
    conf = args.prob_threshold
    flag_fd = args.FDO
    flag_fld = args.FLD
    
    '''Initializing all the classes and checking the different model classes for any unsupported layers'''
    
    Face_Det = Model_fd(model_fd)
    start_lt_fd = time.time()
    Face_Det.load_model(device)
    total_lt_fd = round((time.time() - start_lt_fd), 2)
    Face_Det.check_model(device)
    
    Head_Pose = Model_hpe(model_hpe)
    start_lt_hpe = time.time()
    Head_Pose.load_model(device)
    total_lt_hpe = round((time.time() - start_lt_hpe), 2)
    Head_Pose.check_model(device)
    
    Landmarks_Det = Model_fld(model_fld)
    start_lt_fld = time.time()
    Landmarks_Det.load_model(device)
    total_lt_fld = round((time.time() - start_lt_fld), 2)
    Landmarks_Det.check_model(device)
        
    Gaze_Det = Model_ge(model_ge)
    start_lt_ge = time.time()
    Gaze_Det.load_model(device)
    total_lt_ge = round((time.time() - start_lt_ge), 2)
    Gaze_Det.check_model(device)
    
    mouse = MouseController('medium', 'medium')
    
    '''Reading the input in a loop and passing it through the pipline of all the model's and then using their output
        to move the pointer on screen using the pyautogui python library'''
    
    feed=InputFeeder(input_type=input_type, input_file=input_file)
    feed.load_data()
    (width, height, fps) = feed.get_dim()
    out_video = cv2.VideoWriter(os.path.join('/home/workspace/CPC_project/results/', 'output_video.mp4'), 0x00000021, fps, (width, height))
    
    start_inference_time = time.time()
    counter = 0
    
    for batch in feed.next_batch():
        if np.shape(batch) != ():
            counter+=1
            ppi_fd = Face_Det.preprocess_input(batch)
            outputs_fd = Face_Det.predict(ppi_fd)
            ppo_fd, ymin, ymax, xmin, xmax = Face_Det.preprocess_output(batch, width, height, conf, outputs_fd)

            ppi_hpe = Head_Pose.preprocess_input(ppo_fd)
            (yaw_a, pitch_a, roll_a) = Head_Pose.predict(ppi_hpe)
            (yaw, pitch, roll) = Head_Pose.preprocess_output(yaw_a, pitch_a, roll_a)

            (ppi_fld, height_fd, width_fd) = Landmarks_Det.preprocess_input(ppo_fd)
            outputs_fld = Landmarks_Det.predict(ppi_fld)
            (left_eye, right_eye, batch, ppo_fd_) = Landmarks_Det.preprocess_output(batch, ppo_fd, height_fd, width_fd, outputs_fld, ymin, ymax, xmin, xmax, flag_fld)
            
            if np.shape(left_eye) != () and np.shape(right_eye) != () and np.sum(left_eye) != 0 and np.sum(right_eye) != 0:
                (ppi_ge_left, ppi_ge_right) = Gaze_Det.preprocess_input(left_eye, right_eye)
                outputs_ge = Gaze_Det.predict(ppi_ge_left, ppi_ge_right, yaw, pitch, roll)
                (x,y,z) = Gaze_Det.preprocess_output(outputs_ge)
                print(x,y,z)
                print(counter)
            else:
                continue
                
            (screen_width, screen_height) = pyautogui.size()
            mouse.move(x, y)
            (xx, yy) = pyautogui.position()
            xx = int((width/screen_width)*xx)
            yy = int((height/screen_height)*yy)
            batch[(yy-14):(yy+14),(xx-7):(xx+7)]=[0,0,255]
            
            if flag_fd:
                cv2.rectangle(batch, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
            out_video.write(batch)

        else:
            break
        
        
    feed.close()
    total_time=time.time()-start_inference_time
    total_inference_time=round(total_time, 1)
    fps_avg=counter/total_inference_time

    with open(os.path.join('/home/workspace/CPC_project/results/', 'stats.txt'), 'w') as f:
        f.write(str(total_lt_fd)+'\n')
        f.write(str(total_lt_hpe)+'\n')
        f.write(str(total_lt_fld)+'\n')
        f.write(str(total_lt_ge)+'\n')
        f.write(str(total_inference_time)+'\n')
        f.write(str(fps_avg)+'\n')
    
    print(f"Load_Time-Face-Detection-Model:{total_lt_fd}")
    print(f"Load_Time-Head-Pose-Estimation-Model:{total_lt_hpe}")
    print(f"Load_Time-Facial-Landmarks-Detection-Model:{total_lt_fld}")
    print(f"Load_Time-Gaze-Estimation-Model:{total_lt_ge}")
    print(f"Total_Inference_Time:{total_inference_time}")
    print(f"FPS average:{fps_avg}")
    print(f"Total no. of frames:{counter}")
    
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
    