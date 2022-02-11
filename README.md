# EzEdgeAI  
  
> The version that previously implemented the "Edge Impulse for Linux" Python SDK on the Raspberry Pi 4 is now in the "[concept](https://github.com/on-device-ai/EzEdgeAI/tree/concept)" branch.  
  
This project uses the "Low Code"/"No Code" approach to build a deep learning development environment. It includes a Python componentization framework and a [flow-based programming](https://en.wikipedia.org/wiki/Flow-based_programming) visual editor. The concept is as follows:  
![220211_1](https://user-images.githubusercontent.com/44540872/153596578-665c400e-1d4e-436d-a628-d79644464f24.png)
Each unit of deep learning is componentized to achieve code reuse and to simplify the integration of flow-based programming. Components can be called directly from Python and integrated with the Jupyter Lab environment to achieve the "Low Code" approach. Or integrate components into flow-based visual programming using the [ryvencore-qt](https://github.com/leon-thomm/ryvencore-qt) library to achieve the "No Code" approach.  
This project is still in the early stage of development. Currently, the object detection and inference function of Edge TPU ([Coral](https://coral.ai/products/accelerator/)) is componentized:  
![220211_2](https://user-images.githubusercontent.com/44540872/153594745-185b06f1-7311-4305-a739-8c96de18ba65.png)  
