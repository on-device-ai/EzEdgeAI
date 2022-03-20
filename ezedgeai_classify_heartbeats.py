import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ezedgeai_core import Component
from ezedgeai_core import Input
from ezedgeai_core import Output
from ezedgeai_core import Connection
from ezedgeai_core import Procedure
from ezedgeai_core import UnknownPort
from ezedgeai_core import UnknownProperty
from ezedgeai_core import PortNotConnected

#####

class ClassifyHeartbeatsModelProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )    
    def init( self , component ):
        super( ).init( component )
    def _model( self ):
        # 1D Convolutional Neural Network
        # Ref. : A deep convolutional neural network model to classify heartbeats ( https://www.sciencedirect.com/science/article/pii/S0010482517302810?via%3Dihub )
        inputs  = layers.Input(shape=(260,))
        reshape = layers.Reshape((260,1))(inputs)
        conv1   = layers.Conv1D(5,  3, activation='linear')(reshape)
        conv1   = layers.ReLU()(conv1)
        conv1   = layers.MaxPooling1D(2,2)(conv1)
        conv2   = layers.Conv1D(10, 4, activation='linear')(conv1)
        conv2   = layers.ReLU()(conv2)
        conv2   = layers.MaxPooling1D(2,2)(conv2)
        conv3   = layers.Conv1D(20, 4, activation='linear')(conv2)
        conv3   = layers.ReLU()(conv3)
        conv3   = layers.MaxPooling1D(2,2)(conv3)
        flat    = layers.Flatten()(conv3)
        dense1  = layers.Dense(30, activation='linear')(flat)
        dense1  = layers.ReLU()(dense1)
        dense2  = layers.Dense(20, activation='linear')(dense1)
        dense2  = layers.ReLU()(dense2)
        dense3  = layers.Dense(5, activation='softmax')(dense2)
        model   = keras.Model(inputs=inputs, outputs=dense3)
        # DEBUG
        model.summary()
        return model
    def proc( self ):
        try :
            model_oport = self._component.get_port( 'model' )
            try:
                model_oport.invoke( self._model( ) )
            except PortNotConnected:
                pass
        except UnknownPort:
            pass

class MitBihArrhythmiaDatasetProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )    
    def init( self , component ):
        super( ).init( component )
    def _dataset( self ):
        # MIT-BIH Arrhythmia Database
        # https://physionet.org/content/mitdb/1.0.0/
        mitdbs = [100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234]
        X_list = list() 
        y_list = list()
        for db in mitdbs:
            if os.path.isfile('./beats/'+ str(db) + '.npy') : 
                db_dict = np.load('./beats/'+ str(db) + '.npy',allow_pickle = True).item()
                beats   = db_dict['beats']
                for beat in beats :
                    # X is ECG signal
                    X_list.append(beat[1])
                    # y is categorie
                    y_list.append(beat[0])
            else :
                print('./beats/'+ str(db) + '.npy' + ' file is not exist')
        X = np.array(X_list)
        y = np.array(y_list)
        # DEBUG
        print('Type N number of beats = ' + str(len(np.where(y==0)[0])))
        print('Type S number of beats = ' + str(len(np.where(y==1)[0])))
        print('Type V number of beats = ' + str(len(np.where(y==2)[0])))
        print('Type F number of beats = ' + str(len(np.where(y==3)[0])))
        print('Type Q number of beats = ' + str(len(np.where(y==4)[0])))
        print('Total  number of beats = ' + str(len(y)))
        return ( X, y )
    def proc( self ):
        try :
            dataset_oport = self._component.get_port( 'dataset' )
            try:
                dataset_oport.invoke( self._dataset( ) )
            except PortNotConnected:
                pass
        except UnknownPort:
            pass

class DatasetSplitProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        try :
            dataset_iport = self._component.get_port( 'dataset' )
            test_data_oport = self._component.get_port( 'test_data' )
            train_data_oport = self._component.get_port( 'train_data' )
            dataset = dataset_iport.get_input_port_data( )
            if dataset is not None :
                X = dataset[0]
                y = dataset[1]
                # Create train and test dataset
                X_train , X_test , y_train , y_test = train_test_split( X , y , test_size = 0.1 )
                try:
                    test_data_oport.invoke( ( X_test , y_test ) )
                except PortNotConnected:
                    pass
                try:
                    train_data_oport.invoke( ( X_train , y_train ) )
                except PortNotConnected:
                    pass
        except UnknownPort:
            pass
            
class ForceStopTrainCallback(keras.callbacks.Callback):
    def __init__(self):
        super(ForceStopTrainCallback, self).__init__()
        self._is_force_stop = False
    def force_stop(self):
        self._is_force_stop = True
    def on_train_batch_end(self, batch, logs=None):
        if self._is_force_stop is True :
            # DEBUG
            print('ForceStopTrainCallback::on_train_batch_end() : stop training')
            self.model.stop_training = True
    def on_epoch_end(self, epoch, logs=None):
        if self._is_force_stop is True :
            # DEBUG
            print('ForceStopTrainCallback::on_epoch_end() : stop training')
            self.model.stop_training = True


class ClassifyHeartbeatsModelTrainProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )
        self._model = None
        self._train_data = None
        self._force_stop = None
        self._proc_stop = False
    def init( self , component ):
        super( ).init( component )
    def proc( self ) :
        try :
            model_in_iport = self._component.get_port( 'model_in' )
            train_data_iport = self._component.get_port( 'train_data' )
            model_out_oport = self._component.get_port( 'model_out' )
            model = model_in_iport.get_input_port_data( )
            train_data = train_data_iport.get_input_port_data( )
            if model is not None :
                self._model = model
            elif train_data is not None :
                self._train_data = train_data
            if self._model is not None and self._train_data is not None:
                self._proc_stop = False
                X_train = self._train_data[0]
                y_train = self._train_data[1]
                num_classes = 5
                weight0 = 1.0 - (len(np.where(y_train==0)[0]) / len(y_train))
                weight1 = 1.0 - (len(np.where(y_train==1)[0]) / len(y_train))
                weight2 = 1.0 - (len(np.where(y_train==2)[0]) / len(y_train))
                weight3 = 1.0 - (len(np.where(y_train==3)[0]) / len(y_train))
                weight4 = 1.0 - (len(np.where(y_train==4)[0]) / len(y_train))
                class_weight = {0: weight0 , 1: weight1 , 2: weight2 ,3 : weight3 , 4 : weight4}
                # DEBUG
                print('ClassifyHeartbeatsModelProcedure::proc() : class_weight = ' + str( class_weight ) )
                y_Train = keras.utils.to_categorical( y_train , num_classes )
                y_Train = y_Train.reshape( len(y_Train) , num_classes )
                # Configures the model for training
                self._model.compile( keras.optimizers.Adam() , loss = keras.losses.CategoricalCrossentropy() , metrics = [ keras.metrics.CategoricalAccuracy( ) ] )
                # Stop training when a monitored quantity has stopped improving
                earlyStopping=keras.callbacks.EarlyStopping( monitor = 'val_loss' , patience=3 )
                self._force_stop = ForceStopTrainCallback( )
                # Trains the model for a fixed number of epochs (iterations on a dataset)
                train_history = self._model.fit( X_train , y_Train , batch_size = 32 , epochs = 20 , validation_split = 0.3 , callbacks=[ earlyStopping, self._force_stop ],class_weight = class_weight )
                self._component.set_property( 'train_history' , train_history )
                if self._proc_stop is False :
                    try:
                        model_out_oport.invoke( self._model )
                    except PortNotConnected:
                        pass
                else :
                    # DEBUG
                    print('ClassifyHeartbeatsModelTrainProcedure::proc() : stop training')
                self._model = None
                self._train_data = None
                self._force_stop = None
        except UnknownPort:
            pass
    def stop_proc( self ):
        if self._force_stop is not None :
            self._force_stop.force_stop()
            self._proc_stop = True
            
class TeeProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        try :
            input_iport = self._component.get_port( 'input' )
            output1_oport = self._component.get_port( 'output1' )
            output2_oport = self._component.get_port( 'output2' )
            data = input_iport.get_input_port_data( )
            try:
                output1_oport.invoke( data )
            except PortNotConnected:
                    pass
            try:
                output2_oport.invoke( data )
            except PortNotConnected:
                    pass
        except UnknownPort:
            pass

class ClassifyHeartbeatsModelEvaluteProcedure( Procedure ) :
    def __init__( self , show_confusion_matrix = False ):
        super( ).__init__( )
        self._is_show_confusion_matrix = show_confusion_matrix
        self._model = None
        self._test_data = None
    def init( self , component ):
        super( ).init( component )
    def _show_confusion_matrix( self , confusion_matrix):
        print('\n' + str(confusion_matrix))
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.values.sum() - (FP + FN + TP)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        print('acc  = ' + str(round(ACC[0]*100.0,2)) + '% , ' +           str(round(ACC[1]*100.0,2)) + '% , ' +           str(round(ACC[2]*100.0,2)) + '% , ' +           str(round(ACC[3]*100.0,2)) + '% , ' +           str(round(ACC[4]*100.0,2)) + '% ')
        print('ppv  = ' + str(round(PPV[0]*100.0,2)) + '% , ' +           str(round(PPV[1]*100.0,2)) + '% , ' +           str(round(PPV[2]*100.0,2)) + '% , ' +           str(round(PPV[3]*100.0,2)) + '% , ' +           str(round(PPV[4]*100.0,2)) + '% ')
        print('sen  = ' + str(round(TPR[0]*100.0,2)) + '% , ' +           str(round(TPR[1]*100.0,2)) + '% , ' +           str(round(TPR[2]*100.0,2)) + '% , ' +           str(round(TPR[3]*100.0,2)) + '% , ' +           str(round(TPR[4]*100.0,2)) + '% ')
        print('spec = ' + str(round(TNR[0]*100.0,2)) + '% , ' +           str(round(TNR[1]*100.0,2)) + '% , ' +           str(round(TNR[2]*100.0,2)) + '% , ' +           str(round(TNR[3]*100.0,2)) + '% , ' +           str(round(TNR[4]*100.0,2)) + '% ')
    def proc( self ):
        try :
            model_iport = self._component.get_port( 'model' )
            test_data_iport = self._component.get_port( 'test_data' )
            model = model_iport.get_input_port_data( )
            test_data = test_data_iport.get_input_port_data( )
            if model is not None :
                self._model = model
            elif test_data is not None :
                self._test_data = test_data
            if self._model is not None and self._test_data is not None:
                X_test = self._test_data[0]
                y_test = self._test_data[1]
                num_classes = 5
                y_Test=keras.utils.to_categorical(y_test,num_classes)
                y_Test=y_Test.reshape(len(y_Test),num_classes)
                score = self._model.evaluate(X_test, y_Test, batch_size=32 , verbose=0)
                self._component.set_property( 'score' , score )
                # DEBUG
                print('ClassifyHeartbeatsModelEvaluteProcedure::proc() : Model evaluate score = ' + str( score ))
                if self._is_show_confusion_matrix is True :
                    # Generates output predictions for the input samples
                    prediction = self._model.predict( X_test , batch_size=32 )
                    prediction = prediction.argmax( axis=-1 )
                    confusion_matrix = pd.crosstab( y_test, prediction , rownames=['label'] , colnames=['predict'] )
                    self._show_confusion_matrix(confusion_matrix)
                self._model = None
                self._test_data = None
        except UnknownPort:
            pass

class ConvertToTFLiteModelProcedure( Procedure ) :
    def __init__( self , compile_micro_model = False):
        super( ).__init__( )
        self._compile_micro_model = compile_micro_model
        self._model = None
        self._train_data = None
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        try :
            model_iport = self._component.get_port( 'model' )
            train_data_iport = self._component.get_port( 'train_data' )
            model = model_iport.get_input_port_data( )
            train_data = train_data_iport.get_input_port_data( )
            if model is not None :
                self._model = model
            elif train_data is not None :
                self._train_data = train_data
            if self._model is not None and self._train_data is not None:
                X_train = self._train_data[0]
                # Convert the model to the TensorFlow Lite format with quantization
                converter = tf.lite.TFLiteConverter.from_keras_model( self._model )
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
                train_sub_dataset = X_train[0:100]
                def representative_data_gen():
                    for input_value in train_sub_dataset:
                        yield [input_value.reshape(1,260)]
                converter.representative_dataset = representative_data_gen
                tflite_model = converter.convert( )
                # Save the model to disk
                open("classify_heartbeats.tflite", "wb").write( tflite_model )
                os.system( 'bash -c "xxd -i classify_heartbeats.tflite > classify_heartbeats_cnn_quantized.cc" ' )
                if self._compile_micro_model is True :
                    os.system( 'bash -c "./compiler classify_heartbeats.tflite heartbeats_compiled.cpp heartbeats_" ' )
                self._model = None
                self._train_data = None
        except UnknownPort:
            pass