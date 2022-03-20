# EzEdgeAI  
  
This project uses the "Low Code/No Code" approach to build a deep learning development environment for Edge AI or On-Device AI. It includes a component-based framework and a flow-based visual programming editor. Please refer to the main branch's [README.md](https://github.com/on-device-ai/EzEdgeAI/blob/main/README.md) for the concept of the project's architecture.  
This branch attempts to implement TinyML or TFLite Micro development environments.  It integrates previously developed TinyML related projects such as [OpenM1](https://github.com/on-device-ai/OpenM1), [McuML](https://github.com/on-device-ai/McuML), [tflite-micro-compiler](https://github.com/on-device-ai/tflite-micro-compiler), etc. Use the [ClassifyHeartbeats](https://github.com/on-device-ai/ClassifyHeartbeats) project as an example to demonstrate the TinyML development procedure and deploy the final output to the STM32F746 Discovery kit.  
Please use [EzEdgeAI.ipynb](https://github.com/on-device-ai/EzEdgeAI/blob/tinyml/EzEdgeAI.ipynb) to start the development environment.  
By the way, we can visualize the TinyML development procedure. The Low-Code approach (using the component framework directly) looks like this:  
![220320](https://user-images.githubusercontent.com/44540872/159155021-e687426a-5734-42db-8ae8-9bbbedd68612.png)  
The No-Code approach (flow-based visual programming) looks like this:  
![220316](https://user-images.githubusercontent.com/44540872/159154996-6a471be3-f360-4eae-8305-a2de8333d6e5.png)  