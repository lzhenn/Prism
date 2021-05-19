
# Prism 
![](https://raw.githubusercontent.com/Novarizark/Prism/master/prism.jpg)

Prism is a SOM-based classifier to classify weather types according to regional large-scale signals extracted from wrfout.

### Input Files

#### config.ini
`./conf/config.ini`: Configure file for the model. You may set IO options and trainning details in this file.


#### training data 
`./input/training`: Training data storage, using symbol link to original files. 

#### inference data 
`./input/inference`: inference data storage, using symbol link to original files. 

### Module Files

#### run_build.py
`./run_build.py`: Main script to build Prism using training data. 

#### run_inference.py
`./run_inference.py`: Main script to cast built Prism on inference data. 

#### lib

* `./lib/cfgparser.py`: Module file containing read/write funcs of the `config.ini`

* `./lib/preprocess_wrfinp.py`: Class template to construct the i`wrf_hdl` obj, which contains WRF fields data for classification, such as SLP 

* `./lib/time_manager.py`: Class template to construct time manager obj

#### core 
`./core/prism.py`: Core module, Prism classifier, including train, cast, archive, load method to implement the classifier.

#### utils
`./utils/utils.py`: Commonly used utilities.

#### doc
Documents related to the model.

#### post_process
Scratch script for visualization.

**Any question, please contact Zhenning LI (zhenningli91@gmail.com)**
