'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class Model_fld:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_fld, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.net = None
        self.exec_net = None
        self.plugin = None
        self.input_shape = None
        self.output_shape = None
        self.input_blob = None
        self.output_blob = None
        self.model = model_fld

    def load_model(self, device):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        
        model_structure = self.model+".xml"
        model_weights = self.model+".bin"
        self.plugin = IECore()
        self.net = IENetwork(model_structure, model_weights)
        self.exec_net = self.plugin.load_network(network = self.net, device_name = device, num_requests = 1)
        
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.output_shape = self.net.outputs[self.output_blob].shape

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.exec_net.start_async(request_id = 0, inputs={self.input_blob:image})
        if self.exec_net.requests[0].wait(-1) == 0:
            outputs = self.exec_net.requests[0].outputs[self.output_blob]
        return outputs

    def check_model(self, device):
        sl = self.plugin.query_network(self.net, device_name=device)
        ul = [l for l in self.net.layers.keys() if l not in sl]
        if len(ul) != 0 :
            print('Unsupported layers found:{}'.format(ul))
            print('Check for any extensions for these unsupported layers available for adding to IECore')
            exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        (height_fd, width_fd, channels) = image.shape
        ppi = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LINEAR)
        ppi = np.reshape(ppi, (1, 3, 48, 48))
        return (ppi, height_fd, width_fd)

    def preprocess_output(self, batch, image, height_fd, width_fd, outputs, ymin, ymax, xmin, xmax, flag_fld):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        left_eye_x = int(outputs[0][0] * width_fd)
        left_eye_y = int(outputs[0][1] * height_fd)
        right_eye_x = int(outputs[0][2] * width_fd)
        right_eye_y = int(outputs[0][3] * height_fd)
        
        left_eye_x1 = left_eye_x-40
        left_eye_x2 = left_eye_x+30
        left_eye_y1 = left_eye_y-25
        left_eye_y2 = left_eye_y+25
        right_eye_x1 = right_eye_x-30
        right_eye_x2 = right_eye_x+40
        right_eye_y1 = right_eye_y-25
        right_eye_y2 = right_eye_y+25
        
        left_xmin = xmin + left_eye_x1
        left_ymin = ymin + left_eye_y1
        left_xmax = xmax + left_eye_x2 - width_fd
        left_ymax = ymax + left_eye_y2 - height_fd
        right_xmin = xmin + right_eye_x1
        right_ymin = ymin + right_eye_y1
        right_xmax = xmax + right_eye_x2 - width_fd
        right_ymax = ymax + right_eye_y2 - height_fd
        
        left_eye_xx = xmin + left_eye_x
        left_eye_yy = ymin + left_eye_y
        right_eye_xx = xmin + right_eye_x
        right_eye_yy = ymin + right_eye_y

        if flag_fld:
            batch[(left_eye_yy-7):(left_eye_yy+7),(left_eye_xx-7):(left_eye_xx+7)]=[0,0,255]
            batch[(right_eye_yy-7):(right_eye_yy+7),(right_eye_xx-7):(right_eye_xx+7)]=[0,0,255]
            cv2.rectangle(batch, (left_xmin, left_ymin), (left_xmax, left_ymax), (255,0,0), 2)
            cv2.rectangle(batch, (right_xmin, right_ymin), (right_xmax, right_ymax), (255,0,0), 2)
        
        right_eye = image[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2, :]
        left_eye = image[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2, :]
        return (left_eye, right_eye, batch, image)
        