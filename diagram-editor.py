from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QDialog, QGraphicsTextItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsScene, QGraphicsView, QWidget, QApplication, QHBoxLayout, QListView, QMenu, QPushButton, QVBoxLayout, QLabel, QLineEdit, QFileDialog
import sys

import numpy as np
import cv2
import os

import ezedgeai_core as ez
import ezedgeai_ei_image as ei_img

class Connection:
    """
    - fromPort
    - toPort
    """
    def __init__(self, fromPort, toPort):
        self.fromPort = fromPort
        self.pos1 = None
        self.pos2 = None
        if fromPort:
            self.pos1 = fromPort.scenePos()
            fromPort.posCallbacks.append(self.setBeginPos)
        self.toPort = toPort
        # Create arrow item:
        self.arrow = ArrowItem()
        editor.diagramScene.addItem(self.arrow)
        # Component port connect:
        self.connect = None
    def setFromPort(self, fromPort):
        self.fromPort = fromPort
        if self.fromPort:
            self.pos1 = fromPort.scenePos()
            self.fromPort.posCallbacks.append(self.setBeginPos)
    def getFromPort(self):
        return self.fromPort
    def setToPort(self, toPort):
        self.toPort = toPort
        if self.toPort:
            self.pos2 = toPort.scenePos()
            self.toPort.posCallbacks.append(self.setEndPos)
    def getToPort(self):
        return self.toPort
    def setEndPos(self, endpos):
        self.pos2 = endpos
        self.arrow.setLine(QLineF(self.pos1, self.pos2))
    def setBeginPos(self, pos1):
        self.pos1 = pos1
        self.arrow.setLine(QLineF(self.pos1, self.pos2))
    def delete(self):
        if self.connect is not None and isinstance ( self.connect , ez.Connection ):
            self.connect.disconnect()
        editor.diagramScene.removeItem(self.arrow)

class ParameterDialog(QDialog):
    def __init__(self, parent=None):
        super(ParameterDialog, self).__init__(parent)
        self.button = QPushButton('Ok', self)
        l = QVBoxLayout(self)
        l.addWidget(self.button)
        self.button.clicked.connect(self.OK)
    def OK(self):
        self.close()
        
class RunnerParameterDialog(QDialog):
    def __init__(self, parent=None, component=None):
        super(RunnerParameterDialog, self).__init__(parent)
        self.component = component
        self.model = None
        self.devno = None
        # verticalLayout
        self.verticalLayout = QVBoxLayout(self)
        # horizontalLayout1
        self.horizontalLayout1 = QHBoxLayout(self)
        self.label1 = QLabel(self)
        self.label1.setText('Select .eim model :')
        self.button1 = QPushButton('&Select', self)
        self.button1.clicked.connect(self.SelectEIM)
        self.horizontalLayout1.addWidget(self.label1)
        self.horizontalLayout1.addWidget(self.button1)
        # horizontalLayout2
        self.horizontalLayout2 = QHBoxLayout(self)
        self.label2 = QLabel(self)
        self.label2.setText('Camera Device No. :')
        self.edit2 = QLineEdit(self)
        if self.component is not None:
            self.edit2.setText(str(self.component.get_property('devno')))
        self.edit2.setInputMask('9')
        self.edit2.textChanged.connect(self.editTextChanged)
        self.horizontalLayout2.addWidget(self.label2)
        self.horizontalLayout2.addWidget(self.edit2)
        # horizontalLayout3
        self.horizontalLayout3 = QHBoxLayout(self)
        self.ok_button = QPushButton('&OK', self)
        self.ok_button.clicked.connect(self.OK)
        self.cancel_button = QPushButton('&Cancel', self)
        self.cancel_button.clicked.connect(self.Cancel)
        self.horizontalLayout3.addWidget(self.ok_button)
        self.horizontalLayout3.addWidget(self.cancel_button)
        # add layout
        self.verticalLayout.addLayout(self.horizontalLayout1)
        self.verticalLayout.addLayout(self.horizontalLayout2)
        self.verticalLayout.addLayout(self.horizontalLayout3)
    def SelectEIM(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            head, tail = os.path.split( filenames[0] )
            self.model = tail    
    def editTextChanged(self, text):
        self.devno = text        
    def OK(self):
        if self.component is not None:
            if self.model is not None:
                self.component.set_property('model',self.model)
                # DEBUG
                print(self.component.get_property('model'))
            if self.devno is not None:
                self.component.set_property('devno',int(self.devno))
                # DEBUG
                print(self.component.get_property('devno'))
        self.close()
    def Cancel(self):
        self.close()

class PortItem(QGraphicsEllipseItem):
    """ Represents a port to a subsystem """
    def __init__(self, name, parent=None):
        QGraphicsEllipseItem.__init__(self, QRectF(-6,-6,12.0,12.0), parent)
        self.setCursor(QCursor(QtCore.Qt.CrossCursor))
        # Properties:
        self.setBrush(QBrush(Qt.red))
        # Name:
        self.name = name
        self.posCallbacks = []
        self.setFlag(self.ItemSendsScenePositionChanges, True)
        # Connection:
        self.connection = None
        # BlockItem:
        self.blockItem = parent
    def itemChange(self, change, value):
        if change == self.ItemScenePositionHasChanged:
            for cb in self.posCallbacks:
                cb(value)
            return value
        return super(PortItem, self).itemChange(change, value)
    def mousePressEvent(self, event):
        editor.startConnection(self)
    def portDisconnection(self):
        if self.connection is not None :
            print(self.connection)
            self.connection.delete()
            if self.connection.getFromPort() == self :
                self.connection.getToPort().connection = None
            elif self.connection.getToPort() == self :
                self.connection.getFromPort().connection = None
            self.connection = None

class BlockItem(QGraphicsRectItem):
    """ 
    Represents a block in the diagram
    Has an x and y and width and height
    width and height can only be adjusted with a tip in the lower right corner.

    - in and output ports
    - parameters
    - description
    """
    def __init__(self, name='Untitled', block_item_component=None, parent=None):
        super(BlockItem, self).__init__(parent)
        w = 60.0
        h = 40.0
        # Properties of the rectangle:
        self.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
        self.setBrush(QtGui.QBrush(QtCore.Qt.lightGray))
        self.setFlags(self.ItemIsSelectable | self.ItemIsMovable)
        self.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        # Label:
        self.label = QGraphicsTextItem(name, self)

        self.block_item_component = block_item_component

        # Component:
        self.component = None
        if self.block_item_component is not None :
            self.block_item_component.set_block_item(self)
            self.block_item_component.setup_component()
            # Inputs and outputs of the block:
            self.inputs = []
            self.outputs = []
            self.block_item_component.setup_input_output_port()
        else :
            # Inputs and outputs of the block:
            self.inputs = []
            self.inputs.append( PortItem('a', self) )
            self.inputs.append( PortItem('b', self) )
            self.inputs.append( PortItem('c', self) )
            self.outputs = []
            self.outputs.append( PortItem('y', self) )
            
        # Update size:
        self.changeSize(w, h)
   
    def editParameters(self):
        if self.block_item_component is not None :
            self.block_item_component.exec_parameter_dialog()
        else :
            pd = ParameterDialog(self.window())
            pd.exec_()
     
    def delete(self):
        for iport in self.inputs:
            if iport.connection is not None :
                iport.connection.delete()
                if iport.connection.getFromPort() == iport :
                    iport.connection.getToPort().connection = None
                elif iport.connection.getToPort() == iport :
                    iport.connection.getFromPort().connection = None
                iport.connection = None
        for oport in self.outputs:
            if oport.connection is not None :
                oport.connection.delete()
                if oport.connection.getFromPort() == oport :
                    oport.connection.getToPort().connection = None
                elif oport.connection.getToPort() == oport :
                    oport.connection.getFromPort().connection = None
                oport.connection = None
        editor.diagramScene.removeItem(self)
     
    def changeSize(self, w, h):
        """ Resize block function """
        # Limit the block size:
        if h < 20:
            h = 20
        if w < 40:
            w = 40
        self.setRect(0.0, 0.0, w, h)
        # center label:
        rect = self.label.boundingRect()
        lw, lh = rect.width(), rect.height()
        lx = (w - lw) / 2
        ly = (h - lh) / 2
        self.label.setPos(lx, ly)
        # Update port positions:
        if len(self.inputs) == 1:
            self.inputs[0].setPos(-4, h / 2)
        elif len(self.inputs) > 1:
            y = 5
            dy = (h - 10) / (len(self.inputs) - 1)
            for inp in self.inputs:
                inp.setPos(-4, y)
                y += dy
        if len(self.outputs) == 1:
            self.outputs[0].setPos(w+4, h / 2)
        elif len(self.outputs) > 1:
            y = 5
            dy = (h - 10) / (len(self.outputs) + 0)
            for outp in self.outputs:
                outp.setPos(w+4, y)
                y += dy
        return w, h
                
class ArrowItem(QGraphicsLineItem):
    def __init__(self):
        super(ArrowItem, self).__init__(None)
        self.setPen(QtGui.QPen(QtCore.Qt.red,2))
        self.setFlag(self.ItemIsSelectable, True)

class BlockItemComponent :
    def __init__(self):
        self._block_item = None
    def set_block_item(self, block_item):
        self._block_item = block_item
    def setup_component(self):
        pass
    def setup_input_output_port(self):
        pass
    def exec_parameter_dialog(self):
        pass

class RunnerBlockItemComponent(BlockItemComponent):
    def __init__(self):
        super().__init__()
    def setup_component(self):
        self._block_item.component = ei_img.EiImageRunnerComponent( 'Edge Impulse Image Runner' , ei_img.EiImageRunnerProcedure() )
        self._block_item.component.bind_port( 'output', ez.Output() )
        self._block_item.component.set_property('model','modelfile.eim')
        self._block_item.component.set_property('devno',0)
    def setup_input_output_port(self):
        self._block_item.outputs.append( PortItem('output', self._block_item) )
    def exec_parameter_dialog(self):
        pd = RunnerParameterDialog(self._block_item.window(),self._block_item.component)
        pd.exec_()
        
class ResultBlockItemComponent(BlockItemComponent):
    def __init__(self):
        super().__init__()
    def setup_component(self):
        self._block_item.component = ez.Component( 'Image Classify Result Procedure', None, None, ei_img.ImageClassifyResultProcedure() )
        self._block_item.component.bind_port( 'input', ez.Input())
        self._block_item.component.bind_port( 'output', ez.Output())
    def setup_input_output_port(self):
        self._block_item.inputs.append( PortItem('input', self._block_item) )
        self._block_item.outputs.append( PortItem('output', self._block_item) )
    def exec_parameter_dialog(self):
        pass

class ShowBlockItemComponent(BlockItemComponent):
    def __init__(self):
        super().__init__()
    def setup_component(self):
        self._block_item.component = ez.Component( 'Image Show Procedure', None, None, None )
        self._block_item.component.bind_port( 'input', ez.Input())
    def setup_input_output_port(self):
        self._block_item.inputs.append( PortItem('input', self._block_item) )
    def exec_parameter_dialog(self):
        pass     

class EditorGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        QGraphicsView.__init__(self, scene, parent)
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('component/name'):
            event.accept()
    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat('component/name'):
            event.accept()
    def dropEvent(self, event):
        if event.mimeData().hasFormat('component/name'):
            name = str(event.mimeData().data('component/name'),'utf-8')
            block_item_component = None
            if name == 'Runner':
                block_item_component = RunnerBlockItemComponent()
            elif name == 'Result':
                block_item_component = ResultBlockItemComponent()
            elif name == 'Show':
                block_item_component = ShowBlockItemComponent()
            items = self.scene().items()
            for item in items:
                if type(item) is BlockItem:
                    if block_item_component is not None and type(item.block_item_component) == type(block_item_component) :
                        # DEBUG
                        print(type(item.block_item_component))
                        print(type(block_item_component))
                        block_item_component = None
                        return
            b1 = BlockItem(name,block_item_component)    
            b1.setPos(self.mapToScene(event.pos()))
            self.scene().addItem(b1)
            # DEBUG
            print(self.scene().items())

class LibraryModel(QStandardItemModel):
    def __init__(self, parent=None):
        QStandardItemModel.__init__(self, parent)
    def mimeTypes(self):
        return ['component/name']
    def mimeData(self, idxs):
        mimedata = QMimeData()
        for idx in idxs:
            if idx.isValid():
                txt = self.data(idx, Qt.DisplayRole)
                print(str.encode(txt))
                mimedata.setData('component/name', str.encode(txt))
        return mimedata

class DiagramScene(QGraphicsScene):
    def __init__(self, parent=None):
        super(DiagramScene, self).__init__(parent)
    def mouseMoveEvent(self, mouseEvent):
        editor.sceneMouseMoveEvent(mouseEvent)
        super(DiagramScene, self).mouseMoveEvent(mouseEvent)
    def mouseReleaseEvent(self, mouseEvent):
        editor.sceneMouseReleaseEvent(mouseEvent)
        super(DiagramScene, self).mouseReleaseEvent(mouseEvent)
    def contextMenuEvent(self, event):
        pos = event.scenePos()
        items = self.items(pos)
        for item in items:
            if type(item) is BlockItem:
                menu = QMenu()
                de = menu.addAction('Delete')
                de.triggered.connect(item.delete)
                pa = menu.addAction('Parameters')
                pa.triggered.connect(item.editParameters)
                menu.exec_(event.screenPos())
            if type(item) is PortItem:
                if item.connection is not None :
                    menu = QMenu()
                    di = menu.addAction('Disconnection')
                    di.triggered.connect(item.portDisconnection)
                    menu.exec_(event.screenPos())

class RunnerThread(QtCore.QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self, runner, show, parent=None):
        QThread.__init__(self, parent)
        self.runner = runner
        self.show = show
    def run(self):
        while True:
            self.runner.process()
            if self.show is not None :
                iport = self.show.get_port('input')
                data = iport.get_input_port_data()
                if data is not None :
                    self.change_pixmap_signal.emit(data)
                    
class DiagramEditor(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle("Diagram editor")

        self.proc_run = False
        self.runner_work = None
        
        self.show_w = 320
        self.show_h = 320

        # Widget layout and child widgets:
        self.verticalLayout = QVBoxLayout(self)
        self.runButton = QPushButton()
        self.runButton.setText("Start")
        self.runButton.clicked.connect(self.runButton_clicked)
        self.verticalLayout.addWidget(self.runButton)
        self.horizontalLayout = QHBoxLayout(self)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.image_label = QLabel(self)
        gray = QPixmap(self.show_w,self.show_h)
        gray.fill(QColor('darkGray'))
        self.image_label.setPixmap(gray)
        self.libraryBrowserView = QListView(self)
        self.libraryModel = LibraryModel(self)
        self.libraryModel.setColumnCount(1)
        # Create an icon with an icon:
        pixmap = QPixmap(60, 60)
        pixmap.fill()
        painter = QPainter(pixmap)
        painter.fillRect(10, 10, 40, 40, Qt.blue)
        painter.setBrush(Qt.red)
        painter.drawEllipse(36, 2, 20, 20)
        painter.setBrush(Qt.yellow)
        painter.drawEllipse(20, 20, 20, 20)
        painter.end()

        self.libItems = []
        self.libItems.append( QtGui.QStandardItem(QIcon(pixmap), 'Runner') )
        self.libItems.append( QtGui.QStandardItem(QIcon(pixmap), 'Result') )
        self.libItems.append( QtGui.QStandardItem(QIcon(pixmap), 'Show') )
        for i in self.libItems:
            self.libraryModel.appendRow(i)
        self.libraryBrowserView.setModel(self.libraryModel)
        self.libraryBrowserView.setViewMode(self.libraryBrowserView.IconMode)
        self.libraryBrowserView.setDragDropMode(self.libraryBrowserView.DragOnly)
        self.diagramScene = DiagramScene(self)
        self.diagramView = EditorGraphicsView(self.diagramScene, self)
        self.horizontalLayout.addWidget(self.libraryBrowserView)
        self.horizontalLayout.addWidget(self.diagramView)
        self.horizontalLayout.addWidget(self.image_label)

        self.startedConnection = None
    def startConnection(self, port):
        self.startedConnection = Connection(port, None)
    def sceneMouseMoveEvent(self, event):
        if self.startedConnection:
            pos = event.scenePos()
            self.startedConnection.setEndPos(pos)
    def sceneMouseReleaseEvent(self, event):
        # Clear the actual connection:
        if self.startedConnection:
            pos = event.scenePos()
            items = self.diagramScene.items(pos)
            for item in items:
                if type(item) is PortItem:
                    self.startedConnection.setToPort(item)
            if self.startedConnection.toPort == None:
                self.startedConnection.delete()
            else :
                to_port = item
                from_port = self.startedConnection.getFromPort()
                print(to_port)
                print(from_port)
                if type(to_port) is PortItem and type(from_port) is PortItem and to_port != from_port and to_port.blockItem != from_port.blockItem:
                    if isinstance ( to_port.blockItem.component.get_port(to_port.name) , ez.Input ) :
                        iport = to_port.blockItem.component.get_port(to_port.name)
                        oport = from_port.blockItem.component.get_port(from_port.name)
                    else:
                        iport = from_port.blockItem.component.get_port(from_port.name)
                        oport = to_port.blockItem.component.get_port(to_port.name)
                    self.startedConnection.connect = ez.Connection( oport, iport )
                    self.startedConnection.connect.connect()
                    to_port.connection = self.startedConnection
                    from_port.connection = self.startedConnection
                else :
                    self.startedConnection.delete()
            self.startedConnection = None
    def runButton_clicked(self):
        if self.runner_work is None :
            runner = None
            items = self.diagramView.scene().items()
            for item in items:
                if type(item) is BlockItem and type(item.block_item_component) is RunnerBlockItemComponent :
                    runner = item.component
                    break
            show = None
            items = self.diagramView.scene().items()
            for item in items:
                if type(item) is BlockItem and type(item.block_item_component) is ShowBlockItemComponent :
                    show = item.component
                    break
            if runner is not None :
                self.runner_work = RunnerThread(runner,show)
                self.runner_work.change_pixmap_signal.connect(self.update_image)
        if self.runner_work is not None :
            if self.proc_run is False:
                self.runner_work.start()
                self.runner_work.runner.start_runner_process()
                self.proc_run = True
                self.runButton.setText("Running ...")
    def closeEvent(self, event):
        if self.proc_run is True:
            self.runner_work.runner.stop_runner_process()
            self.proc_run = False            
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        #rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv_img
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.show_w,self.show_h, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

def initDiagramEditor(diagram_editor):
    global editor
    editor = diagram_editor 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    initDiagramEditor(DiagramEditor())
    editor.show()
    editor.resize(1024, 400)
    app.exec_()
