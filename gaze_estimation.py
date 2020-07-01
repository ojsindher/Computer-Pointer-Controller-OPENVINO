'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class Model_ge:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_ge, extensions=None):
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
        self.model = model_ge

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

    def predict(self, left_eye, right_eye, yaw, pitch, roll):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        hpa = np.array([yaw, pitch, roll])
        self.exec_net.start_async(request_id = 0, inputs={'left_eye_image':left_eye, 'right_eye_image':right_eye, 'head_pose_angles':hpa})
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

    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        ppi_left = cv2.resize(left_eye, (60, 60), interpolation=cv2.INTER_AREA)
        ppi_left = np.reshape(ppi_left, (1, 3, 60, 60))
        ppi_right = cv2.resize(right_eye, (60, 60), interpolation=cv2.INTER_AREA)
        ppi_right = np.reshape(ppi_right, (1, 3, 60, 60))
        return (ppi_left, ppi_right)

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        x = outputs[0][0]
        y = outputs[0][1]
        z = outputs[0][2]
        return (x,y,z)
        