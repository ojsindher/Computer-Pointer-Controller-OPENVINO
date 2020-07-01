'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class Model_hpe:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_hpe, extensions=None):
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
        self.model = model_hpe

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
            yaw_a = self.exec_net.requests[0].outputs["angle_y_fc"]
            pitch_a = self.exec_net.requests[0].outputs["angle_p_fc"]
            roll_a = self.exec_net.requests[0].outputs["angle_r_fc"]
        return (yaw_a, pitch_a, roll_a)

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
        ppi = cv2.resize(image, (60, 60))
        ppi = np.reshape(ppi, (1, 3, 60, 60))
        return ppi

    def preprocess_output(self, yaw_a, pitch_a, roll_a):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        yaw = yaw_a[0][0]
        pitch = pitch_a[0][0]
        roll = roll_a[0][0]
        return (yaw, pitch, roll)
