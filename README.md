# EzEdgeAI  
A concept project that uses a low-code/no-code approach to implement deep learning inference on devices. It provides a componentized framework and a visual flow-based programming development environment.  
  
![211130](https://user-images.githubusercontent.com/44540872/143999838-4a8b26e8-ead8-4083-a7b2-8b76e1fcc7d7.png)  
  
This project implemented the "[Edge Impulse for Linux](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux)" [Python SDK](https://docs.edgeimpulse.com/docs/linux-python-sdk) on the [Raspberry Pi 4](https://docs.edgeimpulse.com/docs/raspberry-pi-4) development board and used "[Tutorial: Object Detection](https://docs.edgeimpulse.com/docs/object-detection)" as a demonstration. The operation steps on the Raspberry Pi 4 development board are as follows:  

* Install the Edge Impulse for Linux CLI:  
`curl -sL https://deb.nodesource.com/setup_12.x | sudo bash -`  
`sudo apt install -y gcc g++ make build-essential nodejs sox gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps`  
`npm config set user root && sudo npm install edge-impulse-linux -g --unsafe-perm`  
  
* Connecting to Edge Impulse:  
`edge-impulse-linux --disable-camera`  
> For unknown reasons, the edge-impulse-linux program will not detect that my USB Camera (UVC) is on the Raspberry Pi 4. So add the --disable-camera option to avoid this issue.  
  
* Install the SDK:    
`sudo apt-get install libatlas-base-dev libportaudio0 libportaudio2 libportaudiocpp0 portaudio19-dev`  
`pip3 install edge_impulse_linux -i https://pypi.python.org/simple`  
  
* Update the NumPy:  
`sudo pip3 install -U numpy`  
  
* Download the model file via:  
`edge-impulse-linux-runner --download modelfile.eim`  
and put the model file in the project directory.
  
* Install the PyQt5:  
`sudo apt-get install python3-pyqt5`  
  
* Run the EzEdgeAI project:  
`python diagram-editor.py`
  