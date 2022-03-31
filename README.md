# Introduction
The region of interests (ROIs) extraction is the most fundamental step in analyzing metabolomic dataset acquired by liquid chromatography mass spectrometry (LC-MS).
However, noises and backgrounds existing in LC-MS data often affect the quality of extracted ROIs. 
Therefore, the development of effective ROIs evaluation algorithms is necessary to eliminate the false positives meanwhile keep the false negative rate as low as possible. 
In this study, deep fused filter of ROIs (dffROI) was proposed to improve the accuracy of ROI extraction by combing the handcrafted evaluation metrics with convolution-al neural network (CNN)-learned representations. 
Results show that dffROI can achieve higher accuracy, sensitivity and lower false positive rate. 
The model-agnostic feature importance demon-strates the necessity of fusing handcrafted evaluation metrics with the  convolutional neural network representations. 
DffROI is an automatic, robust and universal method for ROI filtering by virtue of information fusion and end-to-end learn-ing. 
It has been integrated into KPIC2 framework previously proposed by our group to facilitate real metab-olomic LC-MS dataset analysis.
