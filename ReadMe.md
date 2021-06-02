
# Prism 
![](https://raw.githubusercontent.com/Novarizark/Prism/master/prism.jpg)

Prism is a SOM-based classifier to classify weather types according to regional large-scale signals extracted from wrfout.


### Install

Please install python3 using the latest Anaconda3 distribution. [Anaconda3](https://www.anaconda.com/products/individual) with python3.8 has been fully tested, lower version of python3 may also work (without testing).

Now, we recommend to create a new environment in Anaconda and install the `requirements.txt`:

```bash
conda create -n test_prism python=3.8
conda activate test_prism
pip install -r requirements.txt
```

### Usage

Setup necessary parameters in `./conf/config.ini` to link training data, and type

```bash
python3 run_build.py
```

This command will run the training pipeline of Prism, which may take very long time regarding your training sample size. 
If things go smothly, you would expect to see three files `som.archive`, `som_cluster.nc`, and `train_cluster.csv` generated in `./db/`.

For inference pipeline, first link inference data, and type

```bash
python3 run_inference.py
```
If you see the logging streaming yields something like:

```
2021-06-02 16:36:05,780 - INFO : core.prism>>casting...
2021-06-02 16:36:08,030 - INFO : core.prism>>prism inference is completed!
*********************PRISM ACCOMPLISHED*********************
```
Congratulations! You would expect to see `inference_cluster.csv` in the `output` folder. This file archives the clustered result.


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
