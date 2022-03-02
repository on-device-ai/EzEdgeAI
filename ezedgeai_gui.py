import sys
import os
import cv2
import numpy as np
from PIL import Image

os.environ['QT_API'] = 'pyside2'  # tells qtpy to use PySide2
import ryvencore_qt as rc

from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout

from qtpy.QtWidgets import QPushButton, QLabel
from qtpy.QtCore import Qt, Signal, Slot, QEvent, QThread
from qtpy.QtGui import QPixmap,QColor,QImage

from ezedgeai_core import Component
from ezedgeai_core import Input
from ezedgeai_core import Output
from ezedgeai_core import Connection

from ezedgeai_tflite_object_detection import TFLiteModelProcedure
from ezedgeai_tflite_object_detection import CameraImageInputComponent
from ezedgeai_tflite_object_detection import CameraImageInputProcedure
from ezedgeai_tflite_object_detection import TFLiteInterpreterProcedure
from ezedgeai_tflite_object_detection import ObjectDetectionResultProcedure
        
#####

class EdgeTpuModelNode(rc.Node):
    
    title = 'Mobilenet SSD Model'
    # all basic properties
    init_inputs = [
    ]
    init_outputs = [
        rc.NodeOutputBP()
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        self._mobilenet_ssd_model = Component( 'Mobilenet V1 SSD COCO EdgeTPU Model' , None , None , TFLiteModelProcedure( 'ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite' , 'coco_labels.txt' ) )
        self._mobilenet_ssd_model.bind_port( 'model', Output() )
    
    def place_event( self ):
        self._mobilenet_ssd_model.process()
        val = self._mobilenet_ssd_model.get_port( 'model' ).get_data_from_port()
        self.set_output_val(0, val)
        
#####

class EzEdgeAI_ThreadManager :
    _instance = None
    _thread_dict = None
    def __new__( cls, *args, **kwargs): 
        if cls._instance is None: 
            cls._instance = super().__new__(cls) 
        return cls._instance 
    def __init__( self ):
        if self._thread_dict is None :
            self._thread_dict = {}
    def start_thread( self, thread ):
        is_valid_thread = False
        if thread is not None :
            thread_id = str( id( thread ) )
            if thread_id in self._thread_dict :
                is_valid_thread = True
            else :
                if isinstance( thread , QThread ) :
                    self._thread_dict[ thread_id ] = {'thread' : thread , 'thread_start' : False}
                    is_valid_thread = True
                    # DEBUG
                    print('EzEdgeAI_ThreadManager::start_thread() : self._thread_dict = ' + str( self._thread_dict ))
        if is_valid_thread is True :
            ezedgeai_thread = self._thread_dict[ thread_id ]
            if ezedgeai_thread['thread_start'] is False :
                ezedgeai_thread = self._thread_dict[ thread_id ]
                ezedgeai_thread['thread'].start()
                ezedgeai_thread['thread_start'] = True
    def stop_thread( self, thread ):
        if thread is not None :
            thread_id = str( id( thread ) )
            if thread_id in self._thread_dict :
                ezedgeai_thread = self._thread_dict[ thread_id ]
                if ezedgeai_thread['thread_start'] is True : 
                    ezedgeai_thread['thread'].stop()
                    ezedgeai_thread['thread'].quit()
                    ezedgeai_thread['thread'].wait()
                    ezedgeai_thread['thread_start'] = False
    def remove_thread( self, thread ):
        if thread is not None :
            thread_id = str( id( thread ) )
            if thread_id in self._thread_dict :
                ezedgeai_thread = self._thread_dict[ thread_id ]
                self.stop_thread( ezedgeai_thread['thread'] )
                del self._thread_dict[ thread_id ]
                # DEBUG
                print('EzEdgeAI_ThreadManager::remove_thread() : self._thread_dict = ' + str( self._thread_dict ))
    def stop_all_thread( self ):
        for thread_id in self._thread_dict :
          ezedgeai_thread = self._thread_dict[ thread_id ]
          self.stop_thread( ezedgeai_thread['thread'] )
          
#####

class CameraImageInputNode_MainWidget(QPushButton, rc.MWB):
    def __init__(self, params):
        rc.MWB.__init__(self, params)
        QPushButton.__init__(self)
        self.setText('Start')
        self.clicked.connect(self.update_node)

class CameraImageInputNodeThread(QThread):
    def __init__(self, node, parent=None):
        QThread.__init__(self, parent)
        self._node = node
        self._isRunning = True
        self._camera_image_source = CameraImageInputComponent( 'Camera Image Source' , CameraImageInputProcedure( ) )
        self._camera_image_source.bind_port( 'image', Output() )
    def run(self):
        if not self._isRunning:
            self._isRunning = True
        # DEBUG
        print('[Camera] Capture Start')
        self._camera_image_source.capture_start( 0 )
        val = self._camera_image_source.get_port( 'image' ).get_data_from_port()
        self._node.set_output_val(0, val)
        while self._isRunning == True:
            if self._node is not None :
                self._camera_image_source.process( )
                val = self._camera_image_source.get_port( 'image' ).get_data_from_port()
                self._node.set_output_val(0, val)
        # DEBUG
        print('[Camera] Capture Stop')
        self._camera_image_source.capture_stop( )
        val = self._camera_image_source.get_port( 'image' ).get_data_from_port()
        self._node.set_output_val(0, val)
        # DEBUG
        print("finished...")
    def stop(self):
        self._isRunning = False

class CameraImageInputNode(rc.Node):
    
    title = 'Camera Capture'
    main_widget_class = CameraImageInputNode_MainWidget
    main_widget_pos = 'between ports'
    init_inputs = [
    ]
    init_outputs = [
        rc.NodeOutputBP()
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        self._node_thread = None
        self._thread_start = False
        self._thread_manager = EzEdgeAI_ThreadManager( )
        # DEBUG
        print('CameraImageInputNode::__init__() : self._thread_manager id = ' + str(id(self._thread_manager)))
        
    def update_event(self, inp=-1): 
        if self._node_thread is None:
            self._node_thread = CameraImageInputNodeThread(self)
            self._thread_start = False         
        if self._node_thread is not None:
            if self._thread_start is False :
                self._thread_manager.start_thread(self._node_thread)
                self.main_widget().setText('Stop')
                self._thread_start = True
            else :
                self._thread_manager.stop_thread(self._node_thread)
                self.main_widget().setText('Start')
                self._thread_start = False
    def remove_event( self ):
        self._thread_manager.remove_thread( self._node_thread )
        self._node_thread = None
        
#####

class TFLiteInterpreterNode(rc.Node):
    
    title = 'TFLite Interpreter'
    # all basic properties
    init_inputs = [
        rc.NodeInputBP(),
        rc.NodeInputBP()
    ]
    init_outputs = [
        rc.NodeOutputBP()
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        
        self._tflite_interpreter = Component( 'TFLite Interpreter' , None , None , TFLiteInterpreterProcedure( ) )
        self._tflite_interpreter.bind_port( 'model' , Input() )
        self._tflite_interpreter.bind_port( 'image' , Input() )
        self._tflite_interpreter.bind_port( 'result' , Output() )

        self._tflite_interpreter.set_property('threshold',0.7) 

    def update_event(self, inp=-1):
        model = self.input(0)
        self._tflite_interpreter.get_port( 'model' ).invoke(model)
        image = self.input(1)
        self._tflite_interpreter.get_port( 'image' ).invoke(image)
        val = self._tflite_interpreter.get_port( 'result' ).get_data_from_port()
        self.set_output_val(0, val)
        
#####

class DetectionResultNode(rc.Node):
    
    title = 'Object Detection Result'
    # all basic properties
    init_inputs = [
        rc.NodeInputBP()
    ]
    init_outputs = [
        rc.NodeOutputBP()
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        
        self._detection_result =  Component( 'Object Detection Result Process' , None , None , ObjectDetectionResultProcedure( ) )
        self._detection_result.bind_port( 'result' , Input() )
        self._detection_result.bind_port( 'image' , Output() )

    def update_event(self, inp=-1):
        result = self.input(0)
        
        self._detection_result.get_port( 'result' ).invoke(result)
        val = self._detection_result.get_port( 'image' ).get_data_from_port()
        self.set_output_val(0, val)
           
#####

class ImageShowNode_MainWidget(rc.MWB, QLabel):
    def __init__(self, params):
        rc.MWB.__init__(self, params)
        QLabel.__init__(self)

        self._img_show_w = 320
        self._img_show_h = 240

        self.resize(self._img_show_w+4, self._img_show_h+4)
        
        layout = QVBoxLayout()
        self._image_label = QLabel(self)
        gray = QPixmap(self._img_show_w,self._img_show_h)
        gray.fill(QColor('darkGray'))
        self._image_label.setPixmap(gray)
        layout.addWidget(self._image_label)
        self.setLayout(layout)
    
    @Slot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self._image_label.setPixmap(qt_img)
    def convert_cv_qt(self, cv_img):
        # rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv_img
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self._img_show_w,self._img_show_h, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class ImageShowNode(rc.Node):  

    title = 'Image Show'
    init_inputs = [
        rc.NodeInputBP()
    ]
    main_widget_class = ImageShowNode_MainWidget
    main_widget_pos = 'below ports'
    
    change_pixmap_signal = Signal(np.ndarray)  
    
    def __init__(self, params):
        super().__init__(params)
    
    def view_place_event(self):
        self.change_pixmap_signal.connect(self.main_widget().update_image)
    
    def update_event(self, inp=-1):
        image = self.input(0)
        if image is not None and isinstance(image,Image.Image) is True:
            display_image = np.asarray(image)
            self.change_pixmap_signal.emit(display_image)
            
#####

class EzEdgeAI_MainWindow( QMainWindow ):
    def __init__( self ):
        super().__init__()
        self._thread_manager = EzEdgeAI_ThreadManager( )
        # DEBUG
        print('EzEdgeAI_MainWindow::__init__() : self._thread_manager id = ' + str(id(self._thread_manager)))
    def closeEvent(self,event):
        # DEBUG
        print('EzEdgeAI_MainWindow::closeEvent()')
        self._thread_manager.stop_all_thread( )

if __name__ == "__main__":

    # creating the application and a window
    app = QApplication()
    mw = EzEdgeAI_MainWindow()

    # creating the session, registering, creating script
    session = rc.Session()
    session.design.set_flow_theme(name='pure light')
    session.register_nodes([EdgeTpuModelNode, CameraImageInputNode, TFLiteInterpreterNode, DetectionResultNode,ImageShowNode])
    script = session.create_script('EzEdgeAI', flow_view_size=[1000, 800])

    mw.setCentralWidget(session.flow_views[script])

    mw.show()
    sys.exit(app.exec_())
