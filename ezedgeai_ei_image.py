import cv2
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

import multiprocessing as mp

from ezedgeai_core import Component
from ezedgeai_core import Input
from ezedgeai_core import Output
from ezedgeai_core import Connection
from ezedgeai_core import Procedure
from ezedgeai_core import PortNotConnected

def now():
    return round(time.time() * 1000)

def image_runner_process( component ):
    modelfile = component.modelfile
    video_capture_device_id = component.camera_device_id
    
    queue = component.runner_queue

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            
            camera = cv2.VideoCapture(video_capture_device_id)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, video_capture_device_id))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            # limit to ~10 fps here
            next_frame = 0

            for res, img in runner.classifier(video_capture_device_id):
                if (next_frame > now()):
                    time.sleep((next_frame - now()) / 1000)
                
                queue.put((res,img))
                    
                next_frame = now() + 100
        finally:
            if (runner):
                # DEBUG
                print('Runner stop ...')
                runner.stop()
                
def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

class EiImageRunnerComponent( Component ):
    def __init__( self , name , procedure = None ):
        super().__init__( name, None, None, procedure )
        self.runner_queue = None
        self.runner_process = None
        self.modelfile = None
        self.camera_device_id = None
    def _properties_to_parameter( self ):
        name = 'model'
        if name in self._properties :
            model = self._properties[ name ]
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.modelfile = os.path.join(dir_path, model)
        else :
            self.modelfile = ''
        name = 'devno'
        if name in self._properties :
            devno = self._properties[ name ]
            if isinstance ( devno , int):
                self.camera_device_id = devno
        else :
            port_ids = get_webcams()
            if len(port_ids) == 0:
                raise Exception('Cannot find any webcams')
            if len(args)<= 1 and len(port_ids)> 1:
                raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
            self.camera_device_id = int(port_ids[0])
    def start_runner_process( self ):
        if self.runner_process is None and self.runner_queue is None :
            self._properties_to_parameter()
            self.runner_queue = mp.Queue()
            self.runner_process = mp.Process(target=image_runner_process, args=(self,))
            self.runner_process.start()
    def stop_runner_process( self ):
        if self.runner_process is not None and self.runner_queue is not None :
            self.runner_process.terminate()
            self.runner_process.join()
            self.runner_process.close()
            self.runner_process = None
            self.runner_queue.close()
            self.runner_queue = None
    def __del__( self ):
        self.stop_runner_process()
        
class EiImageRunnerProcedure( Procedure ) :
    def __init__( self, queue_get_block = True, queue_get_timeout = None ):
        super( ).__init__( )
        self.queue_get_block = queue_get_block
        self.queue_get_timeout = queue_get_timeout
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        oport = self._component.get_port( 'output' )
        try:
            if self._component.runner_queue is not None :
                data = self._component.runner_queue.get( self.queue_get_block, self.queue_get_timeout )
                oport.invoke( data )
        except PortNotConnected:
            pass
        except mp.queues.Empty:
            pass

class ImageClassifyResultProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        iport = self._component.get_port( 'input' )
        oport = self._component.get_port( 'output' )
        data = iport.get_input_port_data()
        res = data[0]
        img = data[1]
        if "classification" in res["result"].keys():
            print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
            for label in labels:
                score = res['result']['classification'][label]
                print('%s: %.2f\t' % (label, score), end='')
            print('', flush=True)
        elif "bounding_boxes" in res["result"].keys():
            print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
            for bb in res["result"]["bounding_boxes"]:
                print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                if bb['value'] > 0.8 :
                    img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
        try:
            oport.invoke( img )
        except PortNotConnected:
            pass
            
class ImageShowProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        iport = self._component.get_port( 'input' )
        img = iport.get_input_port_data()
        cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
if __name__ == '__main__':
    runner = EiImageRunnerComponent( 'Edge Impulse Image Runner' , EiImageRunnerProcedure() )
    runner.bind_port( 'output', Output() )
    
    runner.set_property('model','modelfile.eim')
    runner.set_property('devno',0)
    
    classify_result = Component( 'Image Classify Result Procedure', None, None, ImageClassifyResultProcedure() )
    classify_result.bind_port( 'input', Input())
    classify_result.bind_port( 'output', Output())
    
    image_show = Component( 'Image Show Procedure', None, None, ImageShowProcedure() )
    image_show.bind_port( 'input', Input())
    
    connect1 = Connection( runner.get_port( 'output' ), classify_result.get_port( 'input' ) )
    connect2 = Connection( classify_result.get_port( 'output' ), image_show.get_port( 'input' ) )
    
    connect1.connect()
    connect2.connect()
    
    runner.start_runner_process()
    while True:
        runner.process( )
        if cv2.waitKey(1) == ord('q'):
            break
    runner.stop_runner_process()
