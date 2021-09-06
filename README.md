# Week 3 Group 2 Coding

## Summary

We explored the Mish, GELU, and ELU papers and explored the CIFAR10, STL10, and Twitter POS tagging datasets. 

## Reference

The CIFAR10 dataset references the following blog: https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54

## CIFAR10

## STL10

### Dataset

We used the STL-10 dataset for another set of experiments. Consisting of 5000 training images and 8000 test images, the STL-10 dataset was acquired via labeled examples on Imagenet. All images are 96x96 and are equally represented.

### Model

The neural network used for training on this dataset consists of 6 convolutional layers and one fully connected layer. The first convolutional layer used a kernel size of 5 and outputted 32 channels from the original 3. The following 4 convolutional layers used a kernel size of 3 and was organized in two blocks where a block was composed of two layers that increased the number of channels by 32 and then kept the channels constant. The final convolutional layer increased the number of channels to 128 before the fully connected layer reduced it back down to 10. Each convolutional layer used a stride of 2.

### Code

The code used to train the model was organized into three main files under the ELU/Model directory. The models.py folder defined the neural network, the utils.py file read in the STL-10 data and performed transformations on the data, and the train.py trained the models and performed evaluations on the validation data. The train.py file also created the ELU/run and ELU/logs directories that recorded data about each run. DownloadSTL.sh used the ELU/Setup directory to download the STL-10 dataset and prep it for use by the utils.py file. Finally, run_experiments.py ran tests described in the ELU/Experiments directory.

### Experiments

All experiments were run using GCP. The virtual machine chosen was a n1-standard-2 (2 vCPUs with 7.5 GB memory) with a NVIDIA Tesla P4 GPU. A Deep Learning Image with Pytorch 1.9 and CUDA 110 was used with 50 GB of disk space. The experiments all tested the performance of the network described above using 5 different activation functions: ReLU, LReLU, Mish, ELU, and GELU. The effect of batch normalization on these activation functions was also tested with a total of 10 samples for each configuration. The main metrics collected were total training time and the accuracy of the model on the validation dataset on the last epoch. All tests were run for 30 epochs with the same hyperparameters and optimizer (SGD optimizer with 0.9 momentum, 0.0005 weight decay, and 64 batch size). The precise details of each run were recorded in the log directory and can be seen by running the first cell in the ELU/TensorboardLogs.ipynb file.

### Results

![STL10 Results](/resources/STL10_Results.png)

When accuracy is considered without batch normalizations, the Mish activation function slightly outperformed the other functions achieving an average of 0.55 (0.025) classifications correct on the validation set for the final epoch. The GELU function, however, performed much worse than the average, achieving an average final accuracy of 0.24 ( 0.38). This result is due to 6 different tests where models with the GELU activation function did not train. Looking at the log files in the ipynb folder shown below, we can see that for some tests the loss did not seem to decrease during the entirety of the test run. The model refused to train and the accuracy on the validation set remained at 10%. It is unknown why this occurred for some tests and why this failed to occur for any runs with batch normalization.

When batch normalization was included in the model, the results from different activation functions become a lot more comparable. No activation function appears to outperform the mean, but the ELU function performs slightly lower than average on the final validation epoch with an mean accuracy of 0.59 (+- 0.010)

![STL10 Time vs. Activation](/resources/STL10_Time_vs_Activation.png)
![STL10 Loss](/resources/STL10_Loss.png)


There were no significant differences in the total time required to train models for 30 epochs due to activation functions. However, as can be seen in the boxplot above, runs with batch normalization were slower on average than runs without batch normalization.

In conclusion, none of the activation functions seem to be significantly better than the other functions for the model and dataset tested. While the Mish activation function performed slightly better when there was batch normalization, the increase in performance was small and unlikely to provide significant benefit. With batch normalization, only the ELU activation function proved to perform worse than the mean and even then only slightly. Finally, no activation function provided a significant benefit to the training speed, the only significant difference in the times of the runs occurring due to batch normalization.

## Twitter POS

### Dataset

Tweet part-of-speech tagging is a natural language processing dataset focusing on tagging words. The dataset is relatively small (1000 training, 327 validation, and 500 testing tweets), which make it suitable for us to study the generalization ability of the trained model.

### Model

According to the paper, we use a simple two-layer network in order to test the generalization ability and convergence speed under different activation functions. The network was trained using 50 epochs, Adam optimizer, CrossEntropyLoss, and learning rates of 1e-3, 1e-4, and 1e-5. 

### Discussion

To help us observe the convergence, we show the loss value and classification error based on the median of three runs.

They follow the settings of:

Learning rate = [1e-2, 1e-3, 1e-4, 1e-5]

![Twitter Training Loss 1](/resources/Twitter_Training_Loss1.png)
![Twitter Training Error 1](/resources/Twitter_Training_Error1.png)

[Classification error (best)]

relu 	 16.765%

prelu 	 15.436%

elu 	 17.827%

silu 	 17.030%

mish 	 17.394%

gelu 	 16.555%

![Twitter Training Loss 2](/resources/Twitter_Training_Loss2.png)
![Twitter Training Error 2](/resources/Twitter_Training_Error2.png)

[Classification error (best)]

relu 	 12.248%

prelu 	 12.304%

elu 	 12.570%

silu 	 12.416%

mish 	 12.472%

gelu 	 12.248%


In the setting lr=1e-3, Elu shows the slowest convergence speed among all the others according to the first figure.

Due to the high learning rate, all methods converage relatively fast and are satured after 10th epochs on classification error. The final classification error indicates that relu and gelu have the best overall accuracy.

We concluded that for a fast learning rate like 1e-3, the model can only retain sub-optimal results and Elu is the most sensitive one to high learning rate.

![Twitter Training Loss 3](/resources/Twitter_Training_Loss3.png)
![Twitter Training Error 3](/resources/Twitter_Training_Error3.png)

[Classification error (best)]

relu 	 12.150%

prelu 	 12.276%

elu 	 13.940%

silu 	 13.255%

mish 	 12.933%

gelu 	 12.458%


![Twitter Training Loss 4](/resources/Twitter_Training_Loss4.png)
![Twitter Training Error 4](/resources/Twitter_Training_Error4.png)

[Classification error (best)]

relu 	 16.261%

prelu 	 15.674%

elu 	 15.646%

silu 	 17.156%

mish 	 16.443%

gelu 	 16.695%

![Twitter Accuracy](/resources/Twitter_Accuracy.png)

In this section, we compare the training and inference time for XX / XX samples, repectively.

Avg. training time per samples for relu ms is 126.2016 μs

Avg. training time per samples for prelu ms is 128.7802 μs

Avg. training time per samples for elu ms is 92.9664 μs

Avg. training time per samples for silu ms is 117.4599 μs

Avg. training time per samples for mish ms is 133.7703 μs

Avg. training time per samples for gelu ms is 214.3286 μs


Avg. inference time per samples for relu ms is 4.5404 μs

Avg. inference time per samples for prelu ms is 7.0469 μs

Avg. inference time per samples for elu ms is 5.1761 μs

Avg. inference time per samples for silu ms is 4.8332 μs

Avg. inference time per samples for mish ms is 10.7620 μs

Avg. inference time per samples for gelu ms is 5.5138 μs


![Twitter Training Time](/resources/Twitter_Training_Time.png)
![Twitter Inference Time](/resources/Twitter_Inference_Time.png)
