class UnknownPort(Exception):
    pass
    
class UnknownProperty(Exception):
    pass

class UnboundPort(Exception):
    pass
    
class PortNotConnected(Exception):
    pass
    
class ConnectionNotConfigured(Exception):
    pass
    
class ImproperConnection(Exception):
    pass
    
class Component :
    def __init__( self , name , ports = None , properties = None , procedure = None ):
        self._name = name
        if ports is None :
            ports = dict ()
        self._ports = ports
        if properties is None :
            properties = dict ()
        self._properties = properties
        self._procedure = procedure
        if self._procedure is not None :
            self._procedure.init( self )
    def bind_port( self , name , ref ):
        self._ports[ name ] = ref
        ref.set_component( self )
    def unbind_port( self , name ):
        if name not in self._ports :
            raise UnknownPort
        self._ports[ name ].set_component( None )
        del self._ports [ name ]
    def process( self ) :
        if self._procedure is not None :
            self._procedure.proc()
    def get_name( self ) :
        return self._name
    def get_port( self , name ):
        if name not in self._ports :
            raise UnknownPort
        return self._ports[ name ]
    def get_property( self , name ):
        if name not in self._properties :
            raise UnknownProperty
        return self._properties[ name ]
    def set_property( self , name , value ):
        self._properties[ name ] = value

class Port :
    def __init__( self ):
        self._data = None
        self._component = None
    def set_component( self , component ):
        self._component = component
    def get_component( self ):
        return self._component
    def invoke( self , data ):
        pass

class Input( Port ):
    def __init__( self ):
        super( ).__init__( )
        self._invoked = False
    def get_input_port_data( self ):
        if self._invoked is True :
            return self._data
        return None        
    def invoke( self , data ):
        self._data = data
        if self._component is None :
            raise UnboundPort
        self._invoked = True
        self._component.process()
        self._invoked = False
 
class Output ( Port ):
    def __init__( self ):
        super( ).__init__( )
        self._ref = None
    def set_ref( self , ref ):
        self._ref = ref
    def get_ref( self ):
        return self._ref
    def invoke( self , data ):
        self._data = data
        if self._ref is None :
            raise PortNotConnected
        self._ref.invoke( data )
    def get_data_from_port( self ):
        data = self._data
        self._data = None
        return data

class Connection :
    def __init__( self , oport , iport ):
        self._input = iport
        self._output = oport
    def connect( self ):
        if not self._input or not self._output :
            raise ConnectionNotConfigured
        if not isinstance ( self._input , Input ) or not isinstance ( self._output , Output ):
            raise ImproperConnection
        self._output.set_ref( self._input )
    def disconnect( self ):
        self._output.set_ref( None )

class Procedure :
    def __init__( self ):
        self._component = None
    def init( self , component ):
        self._component = component
    def proc( self ):
        pass
