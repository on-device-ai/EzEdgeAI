from imutils.video import VideoStream
from tflite_runtime.interpreter import Interpreter, load_delegate
import argparse
import time
import cv2
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import collections
import platform

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

#####

import os

from ezedgeai_core import Component
from ezedgeai_core import Input
from ezedgeai_core import Output
from ezedgeai_core import Connection
from ezedgeai_core import Procedure
from ezedgeai_core import UnknownPort
from ezedgeai_core import UnknownProperty
from ezedgeai_core import PortNotConnected

#####

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.

    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    @property
    def width(self):
        """Returns bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """Returns bounding box height."""
        return self.ymax - self.ymin

    @property
    def area(self):
        """Returns bound box area."""
        return self.width * self.height

    @property
    def valid(self):
        """Returns whether bounding box is valid or not.

        Valid bounding box has xmin <= xmax and ymin <= ymax which is equivalent to
        width >= 0 and height >= 0.
        """
        return self.width >= 0 and self.height >= 0

    def scale(self, sx, sy):
        """Returns scaled bounding box."""
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)

    def translate(self, dx, dy):
        """Returns translated bounding box."""
        return BBox(xmin=dx + self.xmin,
                    ymin=dy + self.ymin,
                    xmax=dx + self.xmax,
                    ymax=dy + self.ymax)

    def map(self, f):
        """Returns bounding box modified by applying f for each coordinate."""
        return BBox(xmin=f(self.xmin),
                    ymin=f(self.ymin),
                    xmax=f(self.xmax),
                    ymax=f(self.ymax))

    @staticmethod
    def intersect(a, b):
        """Returns the intersection of two bounding boxes (may be invalid)."""
        return BBox(xmin=max(a.xmin, b.xmin),
                    ymin=max(a.ymin, b.ymin),
                    xmax=min(a.xmax, b.xmax),
                    ymax=min(a.ymax, b.ymax))

    @staticmethod
    def union(a, b):
        """Returns the union of two bounding boxes (always valid)."""
        return BBox(xmin=min(a.xmin, b.xmin),
                    ymin=min(a.ymin, b.ymin),
                    xmax=max(a.xmax, b.xmax),
                    ymax=max(a.ymax, b.ymax))

    @staticmethod
    def iou(a, b):
        """Returns intersection-over-union value."""
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)


def input_size(interpreter):
    """Returns input image size as (width, height) tuple."""
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    return width, height


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, size, resize):
    """Copies a resized and properly zero-padded image to the input tensor.

    Args:
      interpreter: Interpreter object.
      size: original image size as (width, height) tuple.
      resize: a function that takes a (width, height) tuple, and returns an RGB
        image resized to those dimensions.
    Returns:
      Actual resize ratio, which should be passed to `get_output` function.
    """
    width, height = input_size(interpreter)
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    tensor[:h, :w] = np.reshape(resize((w, h)), (h, w, channel))
    return scale, scale


def output_tensor(interpreter, i):
    """Returns output tensor view."""
    tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
    return np.squeeze(tensor)


def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
    """Returns list of detected objects."""
    boxes = output_tensor(interpreter, 0)
    class_ids = output_tensor(interpreter, 1)
    scores = output_tensor(interpreter, 2)
    count = int(output_tensor(interpreter, 3))

    width, height = input_size(interpreter)
    image_scale_x, image_scale_y = image_scale
    sx, sy = width / image_scale_x, height / image_scale_y

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=float(scores[i]),
            bbox=BBox(xmin=xmin,
                      ymin=ymin,
                      xmax=xmax,
                      ymax=ymax).scale(sx, sy).map(int))

    return [make(i) for i in range(count) if scores[i] >= score_threshold]

#####

def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).

    Args:
      path: path to label file.
      encoding: label file encoding.
    Returns:
      Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return Interpreter(
        model_path=model_file,
        experimental_delegates=[
            load_delegate(EDGETPU_SHARED_LIB,
                          {'device': device[0]} if device else {})
        ])

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        #draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],outline='red')
        draw_rectangle(draw,[(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],'blue',4)
        #draw.text((bbox.xmin + 10, bbox.ymin + 10),'%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),fill='red')
        try:
            font_type = 'NotoSansCJK-Regular.ttc'
            font_size = 18
            font = ImageFont.truetype(font_type, font_size, encoding='utf-8')            
            draw.text((bbox.xmin + 10, bbox.ymin + 10),'%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),fill='blue',font=font)
        except IOError:
            draw.text((bbox.xmin + 10, bbox.ymin + 10),'%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),fill='blue')

#####

class TFLiteModelProcedure( Procedure ) :
    def __init__( self, model = None, labels = None, models_path = None ):
        super( ).__init__( )
        self.model_filename = None
        self.labels_filename = None
        if models_path is None :
            self.models_path = os.getcwd() + '/models/'
        else :
            self.models_path = models_path
        if model is not None and labels is not None :
            self.model_filename = self.models_path + model
            self.labels_filename = self.models_path + labels
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        try :
            model_oport = self._component.get_port( 'model' )
            model_meta = {'model' : self.model_filename, 'labels' : self.labels_filename }
            if self.model_filename is not None and self.labels_filename is not None :
                # DEBUG
                print('TFLiteModelProcedure::proc() model meta data : ' + str(model_meta))
                try:
                    model_oport.invoke(model_meta)
                except PortNotConnected:
                    pass
        except UnknownPort:
            pass
            
class CameraImageInputProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )
        self._video_stream = None
    def init( self , component ):
        super( ).init( component )
    def start( self , camera_device_id ):
        if self._video_stream is None :
            self._video_stream = VideoStream(src=camera_device_id, resolution=(640, 480))
            self._video_stream.start()
            try :
                image_oport = self._component.get_port( 'image' )
                try:
                    # Send command
                    image_oport.invoke( 'start' )
                except PortNotConnected:
                    pass   
            except UnknownPort:
                pass
            time.sleep(1)       
    def stop( self ):
        if self._video_stream is not None :
            self._video_stream.stop()
            self._video_stream = None
            try :
                image_oport = self._component.get_port( 'image' )
                try:
                    # Send command
                    image_oport.invoke( 'stop' )
                except PortNotConnected:
                    pass   
            except UnknownPort:
                pass
            time.sleep(0.1)
    def proc( self ):
        try :
            image_oport = self._component.get_port( 'image' )
            # Read frame from video
            img_cv = self._video_stream.read()
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)
            try:
                image_oport.invoke( image )
            except PortNotConnected:
                pass   
        except UnknownPort:
            pass
    
class CameraImageInputComponent( Component ):
    def __init__( self , name , procedure = None ):
        super().__init__( name, None, None, procedure )
    def capture_start( self , camera_device_id = 0 ):
        if self._procedure is not None and isinstance( self._procedure , CameraImageInputProcedure ) is True:
            self._procedure.start( camera_device_id )
    def capture_stop( self ):
        if self._procedure is not None and isinstance( self._procedure , CameraImageInputProcedure ) is True:
            self._procedure.stop( )
    def __del__( self ):
        self.capture_stop()
        
class TFLiteInterpreterProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )
        self._labels = None
        self._interpreter = None
        self._threshold = 0.4
        self._model_meta = None
    def init( self , component ):
        super( ).init( component )
    def _check_model_meta( self, src, dist ) :
        if src is not None :
            if isinstance(src,dict) is True:
                if 'model' in src.keys() and 'labels' in src.keys():
                    pass
                else :
                    return False
            else :
                return False
            if dist is None :
                return True
            else :
                if src.get('model') == dist.get('model') and src.get('labels') == dist.get('labels'):
                    pass
                else:
                    return True
        return False
    def proc( self ):
        try :
            model_iport = self._component.get_port( 'model' )
            image_iport = self._component.get_port( 'image' )
            result_oport = self._component.get_port( 'result' )
            model_meta = model_iport.get_input_port_data()
            image_data = image_iport.get_input_port_data()
            if model_meta is not None :
                if self._check_model_meta(model_meta, self._model_meta)is True :
                    self._model_meta = model_meta
                    # DEBUG
                    print('TFLiteInterpreterProcedure::proc() self._model_meta = ' + str(self._model_meta))
            elif image_data is not None :
                if isinstance( image_data, str) is True :
                    command = image_data
                    # DEBUG
                    print( 'TFLiteInterpreterProcedure::proc() command : ' + command )
                    if self._labels is None or self._interpreter is None :
                        # DEBUG
                        print('TFLiteInterpreterProcedure::proc() model meta data : ' + str(self._model_meta))
                        if command == 'start' and self._model_meta is not None :
                            self._labels = load_labels(self._model_meta.get('labels'))
                            self._interpreter = make_interpreter(self._model_meta.get('model'))
                            self._interpreter.allocate_tensors()
                            property_name = 'threshold'
                            try :
                                if isinstance( self._component.get_property( property_name ) , float ) is True:
                                    self._threshold = self._component.get_property( property_name )
                            except UnknownProperty:
                                pass
                            self._model_meta = None
                    else :
                        if command == 'stop' :
                            self._labels = None
                            self._interpreter = None
                elif isinstance( image_data, Image.Image) is True :
                    if self._labels is not None and self._interpreter is not None :
                        scale = set_input(self._interpreter, image_data.size,
                                          lambda size: image_data.resize(size, Image.ANTIALIAS))
                        self._interpreter.invoke()
                        objs = get_output(self._interpreter, self._threshold, scale)
                        result_data = { 'image' : image_data, 'results' : objs , 'labels' : self._labels }
                        try:
                            result_oport.invoke( result_data )
                        except PortNotConnected:
                            pass               
        except UnknownPort:
            pass

class ObjectDetectionResultProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )    
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        try :
            result_iport = self._component.get_port( 'result' )
            image_oport = self._component.get_port( 'image' )
            result_data = result_iport.get_input_port_data()
            if result_data is not None and isinstance(result_data, dict) is True :
                image = result_data.get('image')
                results = result_data.get('results')
                labels = result_data.get('labels')
                draw_objects(ImageDraw.Draw(image), results, labels)
                try:
                    image_oport.invoke( image )
                except PortNotConnected:
                    pass
        except UnknownPort:
            pass

class ImageShowProcedure( Procedure ) :
    def __init__( self ):
        super( ).__init__( )    
    def init( self , component ):
        super( ).init( component )
    def proc( self ):
        try :
            image_iport = self._component.get_port( 'image' )
            image = image_iport.get_input_port_data()
            if image is not None and isinstance(image, Image.Image) is True :
                img_rgb = np.asarray(image)
                img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow('Coral Live Object Detection', img_cv)
        except UnknownPort:
            pass
