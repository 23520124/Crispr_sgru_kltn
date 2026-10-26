Crispr-SGRU: Prediction of CRISPR/Cas9 off-target activities with mismatches and indels using stacked BiGRU
## Environment
[CUDA](https://developer.nvidia.com/cuda-toolkit) is necessary for training model in GPU：
CUDA Version:11.8<br>
<br>
The Python packages should be installed :<br>
* [Keras](https://keras.io/) 2.4.3
* [tensorflow-gpu](https://www.tensorflow.org/install/pip) 2.5.0
* [scikit-learn](https://scikit-learn.org/stable/) 0.24.2
## File description
* Train directory:Include the encoding and training process of the model.<br>
* Leave one sgRNA out：Randomly select all samples of one type of sgRNA as the test set, and set a fixed sampling standard during selection: the selected sgRNA set must contain both positive and negative samples. Use all remaining sgRNA samples as the training set.
* Encoder_sgRNA_off.py: Used for Encoding the data from datasets.
* MODEL.py: All models used in the experimental process. 
* weights directory: The weight for the Crisp-SGRU model on all datasets.
* DeepShap: Include Teacher model,Student moodel and knowledge distillation process. 
## Testing 
python model_test.py: Running this file to evaluate the model. (Include loading model weight and datasets, demonstrate model performance of six metrics)<br>
## Datasets 
Include 8 public datasets:
* I1
* I2
* CHANGE-seq
* HEK293t
* K562
* BE3
* II5
* II6
