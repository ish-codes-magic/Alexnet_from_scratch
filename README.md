## AlexNet implementation from scratch

This repository contains an implementation of the AlexNet Convolutional Neural Network from scratch from the [paper](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) 


The code is written in Python and uses PyTorch for the neural network implementation.

Structure

The repository is structured as follows:
```bash
.
│   model.py
│   train.ipynb
│   train.py
│   utils.py
```



[model.py](model.py): This file contains the implementation of the AlexNet model. The model is defined in the AlexNetModel class.

[train.py](train.py): This file contains the training script for the AlexNet model. It uses the model_num_classes, pretrained_model_weights_path, and resume_model_weights_path variables.

[utils.py](utils.py): This file contains utility functions for the project. It includes functions for loading the model_state_dict and the model_weights_path.


### Usage


To train the model, run the ```train.py``` script. You can adjust the number of classes in the model by modifying the ```model_num_classes``` variable in ```train.py```.

If you want to use pretrained weights, you can specify the path to the weights file in the ```pretrained_model_weights_path``` variable in ```train.py```.

If you want to resume training from a specific point, you can specify the path to the weights file in the ```resume_model_weights_path``` variable in ```train.py```.

### Requirements

Python 3.6 or later
PyTorch 1.0 or later

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
