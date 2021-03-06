# EzEdgeAI  
  
> It is an experimental project using the architecture described in this article to implement different applications for study.  
> The version that previously implemented "Edge Impulse for Linux" Python SDK on the Raspberry Pi 4 is in the "[concept](https://github.com/on-device-ai/EzEdgeAI/tree/concept)" branch.  
> The version that previously implemented Edge TPU (Coral) object detection inference is in the "[coral](https://github.com/on-device-ai/EzEdgeAI/tree/coral)" branch.  
> The version that currently implements the TinyML development environment is in the "[tinyml](https://github.com/on-device-ai/EzEdgeAI/tree/tinyml)" branch.  
  
This project uses the "Low Code/No Code" approach to build a deep learning development environment for Edge AI or On-Device AI. It includes a component-based framework and a flow-based visual programming editor. The concept is as follows:  
![220321](https://user-images.githubusercontent.com/44540872/159204182-43c1aa27-e431-4fa8-a722-c69e64c8dfb4.png)  
Deep learning procedures can be transformed into components to achieve code reuse and simplify the integration of flow-based programming. These components can be called directly from Python and integrated with the Jupyter Lab environment for a "low-code" approach. Or integrate these components into flow-based visual programming using the [ryvencore-qt](https://github.com/leon-thomm/ryvencore-qt) library to achieve the "No Code" approach.  