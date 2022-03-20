import sys
import os

os.environ['QT_API'] = 'pyside2'  # tells qtpy to use PySide2
import ryvencore_qt as rc

from qtpy.QtWidgets import QMainWindow, QApplication

from qtpy.QtCore import QThread

from ezedgeai_core import Component
from ezedgeai_core import Input
from ezedgeai_core import Output
from ezedgeai_core import Connection

from ezedgeai_classify_heartbeats import ClassifyHeartbeatsModelProcedure
from ezedgeai_classify_heartbeats import MitBihArrhythmiaDatasetProcedure
from ezedgeai_classify_heartbeats import DatasetSplitProcedure
from ezedgeai_classify_heartbeats import ClassifyHeartbeatsModelTrainProcedure
from ezedgeai_classify_heartbeats import ClassifyHeartbeatsModelEvaluteProcedure
from ezedgeai_classify_heartbeats import ConvertToTFLiteModelProcedure
        
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

class ClassifyHeartbeatsModelNode(rc.Node):
    
    title = 'Model'
    # all basic properties
    init_inputs = [
    ]
    init_outputs = [
        rc.NodeOutputBP()
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        self._model = Component( 'Classify Heartbeats Model' , None , None , ClassifyHeartbeatsModelProcedure( ) )
        self._model.bind_port( 'model', Output() )
    
    def place_event( self ):
        self._model.process()
        val = self._model.get_port( 'model' ).get_data_from_port()
        self.set_output_val(0, val)
        
    def remove_event( self ):
        self._model = None
        
class MitBihArrhythmiaDatasetNode(rc.Node):
    
    title = 'Dataset'
    # all basic properties
    init_inputs = [
    ]
    init_outputs = [
        rc.NodeOutputBP()
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        self._dataset = Component( 'MIT-BIH Arrhythmia Dataset' , None , None , MitBihArrhythmiaDatasetProcedure( ) )
        self._dataset.bind_port( 'dataset', Output() )
    
    def place_event( self ):
        self._dataset.process()
        val = self._dataset.get_port( 'dataset' ).get_data_from_port()
        self.set_output_val(0, val)

    def remove_event( self ):
        self._dataset = None
        
class DatasetSplitNode(rc.Node):
    
    title = 'Split'
    # all basic properties
    init_inputs = [
        rc.NodeInputBP()
    ]
    init_outputs = [
        rc.NodeOutputBP(),
        rc.NodeOutputBP()
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        
        self._split = Component( 'Dataset Split' , None , None , DatasetSplitProcedure( ) )
        self._split.bind_port( 'dataset' , Input() )
        self._split.bind_port( 'train_data' , Output() )
        self._split.bind_port( 'test_data' , Output() )

    def update_event(self, inp=-1):
        dataset = self.input(0)
        self._split.get_port( 'dataset' ).invoke( dataset )
        val = self._split.get_port( 'train_data' ).get_data_from_port()
        self.set_output_val(0, val)
        val = self._split.get_port( 'test_data' ).get_data_from_port()
        self.set_output_val(1, val)
      
class ClassifyHeartbeatsModelTrainComponent(Component):
    def stop_process( self ):
        self._procedure.stop_proc()
      
class ClassifyHeartbeatsModelTrainThread(QThread):
    def __init__(self, node, parent=None):
        QThread.__init__(self, parent)
        self._node = node
        self._model_train = ClassifyHeartbeatsModelTrainComponent( 'Classify Heartbeats Model Train' , None , None , ClassifyHeartbeatsModelTrainProcedure( ) )
        self._model_train.bind_port( 'model_in' , Input() )
        self._model_train.bind_port( 'train_data' , Input() )
        self._model_train.bind_port( 'model_out' , Output() )  
    def run(self):
        # DEBUG
        print('ClassifyHeartbeatsModelTrainThread::run() : Start')
        model_in = self._node.input(0)
        train_data = self._node.input(1)
        if model_in is not None and train_data is not None :
            self._model_train.get_port( 'model_in' ).invoke( model_in )
            self._model_train.get_port( 'train_data' ).invoke( train_data )
            val = self._model_train.get_port( 'model_out' ).get_data_from_port()
            self._node.set_output_val(0, val)
        # DEBUG
        print('ClassifyHeartbeatsModelTrainThread::run() : Finished')
    def stop(self):
        self._model_train.stop_process()
 
class ClassifyHeartbeatsModelTrainNode(rc.Node):
    
    title = 'Model Train'
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
        self._node_thread = None
        self._thread_start = False
        self._thread_manager = EzEdgeAI_ThreadManager( )
        # DEBUG
        print('CameraImageInputNode::__init__() : self._thread_manager id = ' + str(id(self._thread_manager)))
    def update_event(self, inp=-1):
        if self._node_thread is None:
            self._node_thread = ClassifyHeartbeatsModelTrainThread(self)
        if self._node_thread is not None:
            if self._thread_start is True :
                self._thread_manager.stop_thread(self._node_thread)
                self._thread_start = False
            self._thread_manager.start_thread(self._node_thread)
            self._thread_start = True
    def remove_event( self ):
        self._thread_manager.remove_thread( self._node_thread )
        self._node_thread = None
                
class ClassifyHeartbeatsModelEvaluteNode(rc.Node):
    
    title = 'Model Evalute'
    # all basic properties
    init_inputs = [
        rc.NodeInputBP(),
        rc.NodeInputBP()
    ]
    init_outputs = [
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        
        self._model_evaluate = Component( 'Classify Heartbeats Model Evalute' , None , None , ClassifyHeartbeatsModelEvaluteProcedure( show_confusion_matrix=True ) )
        self._model_evaluate.bind_port( 'model', Input())
        self._model_evaluate.bind_port( 'test_data', Input())
        
    def update_event(self, inp=-1):
        model = self.input(0)
        test_data = self.input(1)
        if model is not None and test_data is not None :
            self._model_evaluate.get_port( 'model' ).invoke( model )
            self._model_evaluate.get_port( 'test_data' ).invoke( test_data )
      
class ConvertToTFLiteModelNode(rc.Node):
    
    title = 'Model Convert'
    # all basic properties
    init_inputs = [
        rc.NodeInputBP(),
        rc.NodeInputBP()
    ]
    init_outputs = [
    ]
    color = '#fcba03'
    
    def __init__(self, params):
        super().__init__(params)
        
        self._model_converte = Component( 'Convert To TFLite Model' , None , None , ConvertToTFLiteModelProcedure( compile_micro_model=True ) )
        self._model_converte.bind_port( 'model', Input())
        self._model_converte.bind_port( 'train_data', Input())
        
    def update_event(self, inp=-1):
        model = self.input(0)
        train_data = self.input(1)
        if model is not None and train_data is not None :
            self._model_converte.get_port( 'model' ).invoke( model )
            self._model_converte.get_port( 'train_data' ).invoke( train_data )
     
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
    session.register_nodes([ClassifyHeartbeatsModelNode, MitBihArrhythmiaDatasetNode, DatasetSplitNode, ClassifyHeartbeatsModelTrainNode ,ClassifyHeartbeatsModelEvaluteNode, ConvertToTFLiteModelNode])
    script = session.create_script('EzEdgeAI', flow_view_size=[1000, 800])

    mw.setCentralWidget(session.flow_views[script])

    mw.show()
    sys.exit(app.exec_())
