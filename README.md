# 1.Introduction
The region of interests (ROIs) extraction is the most fundamental step in analyzing metabolomic dataset acquired by liquid chromatography mass spectrometry (LC-MS).
However, noises and backgrounds existing in LC-MS data often affect the quality of extracted ROIs. 
Therefore, the development of effective ROIs evaluation algorithms is necessary to eliminate the false positives meanwhile keep the false negative rate as low as possible. 
In this study, deep fused filter of ROIs (dffROI) was proposed to improve the accuracy of ROI extraction by combing the handcrafted evaluation metrics with convolution-al neural network (CNN)-learned representations. 
Results show that dffROI can achieve higher accuracy, sensitivity and lower false positive rate. 
The model-agnostic feature importance demon-strates the necessity of fusing handcrafted evaluation metrics with the  convolutional neural network representations. 
DffROI is an automatic, robust and universal method for ROI filtering by virtue of information fusion and end-to-end learn-ing. 
It has been integrated into KPIC2 framework previously proposed by our group to facilitate real metab-olomic LC-MS dataset analysis.
![image](https://github.com/zhanghailiangcsu/dffROI/blob/main/TOC.jpg)
# 2.Depends
[Anaconda](https://www.anaconda.com) for python 3.6  
TensorFlow 2.4.0  
[R 4.0.2](https://mirrors.tuna.tsinghua.edu.cn/CRAN)
# 3.Install
1. Install Anaconda  
2. Install [Git](https://git-scm.com/downloads)  
3. Install R 4.0.2  
4. Open commond line, create environment and enter with the following commands:
```
conda create -n dffROI python=3.6
conda activate dffROI
```
5. Clone the repository and enter:
```
git clone https://github.com/zhanghailiangcsu/dffROI.git
cd dffROI
```
6. Install dependency with the following commands
```
pip install -r requirements.txt
```
First you need to install R language. The R language version should be 4.0.2.
Then install KPIC2. The method of installing KPIC2 can refer to https://github.com/hcji/KPIC2.
Next, you need to install the rpy2 package in anaconda.You can install rpy2 using pip. 
```
pip install rpy2
```
Then configure the environment variables of rpy2. Add two new environment variables named R_HOME and R_User to the local environment variables.
The value of R_HOME is the installation location of the R language.
The value of R_User is the installation location of the rpy2 package.
After configuring the environment variables, you can use the python language to call R.
That is, you can use dffROI+KPIC2 to process your data.

# 4.Usage
The dffROI is public at [homepage](https://github.com/zhanghailiangcsu/dffROI), every user can download and use it.
All ROIs can be input into dffROI for processing to filter false positives in the ROIs.  
We provide an example of processing real samples using dffROI+KPIC2.
It is named example(.ipynb) and we upload it.
User can refer to it for processing data.
