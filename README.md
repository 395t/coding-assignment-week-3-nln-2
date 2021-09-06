# Week 3 Group 2 Coding

## Summary

We explored the Mish, GELU, and ELU papers and explored the CIFAR10, STL10, and Twitter POS tagging datasets. 

## Reference

The CIFAR10 dataset references the following [blog](https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54).

The helper functions for Twitter-POS dataset are heavily based on [this repo](https://github.com/hendrycks/GELUs/blob/master/twitter_pos.py).

## CIFAR10

### Model and Dataset

We used the CIFAR-10 dataset. It consists of 60,000 32 x 32 images. The images belong to 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck) and each class is equally represented. 

We altered the model and workflow described in the PyTorch guide (https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54). The model was trained for 25 epochs using the Mish, ELU, and GELU activation functions for varying batch sizes and learning rates. 

The model includes a convolutional layer with an input of 3, output of 6, and a kernel of 5, followed by a max pooling layer, followed by another convolutional layer with an input of 6, output of 16, and kernel size of 5. Then there are three linear layers of input and output size 400 x 120, 120 x 84, and 84 x 10. 

### Code

The experiments were run from the notebooks/cifar10.ipynb notebook. There are three different models defined for using the different activation functions (Mish, ELU, and GELU). The number of epochs and learning rate can be modified as parameters to the fit method. The batch size can be modified thorugh the batch_size variable. 

The code was also transfered to a python file. In the python file, only a single model is defined, so in order to change the activation function, you must change line 109. 

### Performance

We examined the convergence speed and performance by looking at the accuracy and loss across multiple epochs. We also compared the effect of different learning rates. Lastly, we noted the training and inference times. 

Note: For the combined figures, I realized after the fact that a legend was not included. It is as follows:

Blue - Mish 

Green - ELU

Red - GELU

We first varied the batch size and trained 25 epochs with a learning rate of 1e-1. 

### Batch size 40:

Mish: <br />
Val Accuracy - 0.5604 <br />
Val Loss - 1.506 <br />
Training Time - 8 minutes 13 seconds <br />
Inference Time - 1.9 seconds <br />

![Batch 40 Mish](/resources/batch_40_mish.png)

ELU: <br />
Val Acccuracy - 0.5659 <br />
Val Loss - 1.858 <br />
Training Time - 7 minutes 17 seconds <br />
Inference Time - 2.1 seconds <br />

![Batch 40 Mish](/resources/batch_40_mish.png)

GELU: <br />
Val Accuracy - 0.5715 <br />
Val Loss - 1.522 <br />
Training Time - 7 minutes 48 seconds <br />
Inference Time - 2.0 seconds <br />

![Batch 40 Mish](/resources/batch_40_mish.png)

### Batch size 24:

Mish: <br />
Val Accuracy - 0.5709 <br />
Val Loss - 1.5047 <br />
Training Time - 12 minutes 15 seconds <br />
Inference Time - 2.2 seconds<br />

![Batch 24 Mish](/resources/batch_24_mish.png)

ELU <br />
Val Accuracy - 0.5965 <br />
Val Loss - 1.2927 <br />
Training Time - 9 minutes 24 seconds <br />
Inference Time - 2.4 seconds<br />

![Batch 24 Mish](/resources/batch_24_mish.png)

GELU <br />
Val Accuracy - 0.5824 <br />
Val Loss - 1.2347 <br />
Training Time - 10 minutes 42 seconds <br />
Inference Time - 1.8 seconds

![Batch 24 Mish](/resources/batch_24_mish.png)

### Batch size 64:
Mish: <br />
Val Accuracy - 0.5304 <br />
Val Loss - 1.9593 <br />
Training Time - 10 minutes 5 seconds  <br />
Inference Time - 2.4 seconds<br />

![Batch 64 Mish](/resources/batch_64_mish.png)

ELU: <br />
Val Accuracy - 0.6011 <br />
Val Loss - 1.6392 <br />
Training Time - 7 minutes 33 seconds <br />
Inference Time - 2.6 seconds <br />

![Batch 64 Elu](/resources/batch_64_elu.png)

GELU: <br />
Val Accuracy - 0.4926 <br />
Val Loss - 1.9418 <br />
Trainng Time - 8 minutes 5 seconds <br />
Inference Time - 1.5 seconds <br />

![Batch 64 Gelu](/resources/batch_64_gelu.png)

### Batch size 128:
Mish: <br />
Val Accuracy - 0.5901 <br />
Val Loss - 1.3024 <br />
Training Time - 9 minutes 28 seconds <br />
Inference Time - 1.6 seconds <br />


ELU: <br />
Val Accuracy - 0.5956 <br />
Val Loss - 1.4306 <br />
Training Time - 6 minutes 55 seconds <br />
Inference Time - 1.7 seconds <br />


GELU: <br />
Val Accuracy - 0.5777 <br />
Val Loss - 1.3086 <br />
Training Time - 7 minutes 34 seconds <br />
Inference Time - 1.9 seconds <br />

![1e1 Figures](/resources/1e1_figures.png)

We also varied learning rates out of 1e-1, 1e-2, 1e-3, 1e-4 using a batch size of 128.

### 1e-1:
Mish: <br />
Val Accuracy - 0.5901 <br />
Val Loss - 1.3024 <br />

ELU: <br />
Val Accuracy - 0.5956 <br />
Val Loss - 1.4306 <br />

GELU: <br />
Val Accuracy - 0.5777 <br />
Val Loss - 1.3086 <br />

![1e1 Figures](/resources/1e1_figures.png)

### 1e-2:
Mish: <br />
Val Accuracy - 0.4851 <br />
Val Loss - 1.087 <br />

ELU: <br />
Val Accuracy - 0.5574 <br />
Val Loss - 1.2601 <br />

GELU: <br />
Val Accuracy - 0.4644 <br />
Val Loss - 1.5121 <br />

![1e2 Loss](/resources/1e2_loss.png)
![1e2 Accuracy](/resources/1e2_accuracy.png)

### 1e-3:
Mish: <br />
Val Accuracy - .1204 <br />
Val Loss - 2.302 <br />

ELU: <br />
Val Accuracy - 0.2295 <br />
Val Loss - 2.1170 <br />

GELU: <br />
Val Accuracy - 0.1020: <br />
Val Loss - 2.3023 <br />

![1e3 Loss](/resources/1e3_loss.png)
![1e3 Accuracy](/resources/1e3_accuracy.png)

### 1e-4:
Mish: <br />
Val Accuracy - 0.1070 <br />
Val Loss - 2.3042 <br />

ELU: <br />
Val Accuracy - 0.1020 <br />
Val Loss - 2.3031 <br />

GELU: <br />
Val Accuracy - 0.1020 <br />
Val Loss - 2.3030 <br />

![1e4 Loss](/resources/1e4_loss.png)
![1e4 Accuracy](/resources/1e4_accuracy.png)

### Analysis

It appears that the accuracy and loss begin to converge around the 5th epoch across all batch sizes and activation functions. For learning rates smaller than 1e-1, the point of convergence goes back to later epochs, where it appears to be closer to the 10th epoch. For all batch sizes, no one activation function consistently performed better than the rest; however, they all performed exceptionally well with a batch size of 128, where accuracy was close to 60%. With learning rates of 1e-3 and 1e-4, GELU and Mish struggled to learn, leading me to give a recommendation to use ELU in such cases. Across several other cases as well, ELU outperformed the other two activation functions by 1-2%, leading me to stay with this recommendation. In general, across all experiments, the accuracy fell in the range of 50%-60%, where a random guess with 10 classes that are equally represented would be successful 10% of the time. Inference time was about the same across all experiments, and training time varied between 5 to 10 minutes. However, differences in training time may also be due to issues with the computer the models were tranined on. 

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

Tweet part-of-speech tagging is a natural language processing dataset focusing on tagging words. The dataset is relatively small (1000 training, 327 validation, and 500 testing tweets), which makes it suitable for us to study the generalization ability of the trained model.

### Code

The code is orgainzed as follows:
- Notebook for Twitter-POS dataset: `./notebooks/twitter_pos.ipynb`
- Source code for Twitter-POS dataset: `./src/twitter_pos.py` (run by the command `python twitter_pos.py`)
- Dataset for Twitter-POS: `./src/twitter_pos/data/Tweets`

### Model

According to the paper, we use a simple two-layer network in order to test the generalization ability and convergence speed under different activation functions. 

The network was trained using 50 epochs, Adam optimizer, CrossEntropyLoss, and learning rates of 1e-3, 1e-4, and 1e-5. 

Also, we applied dropout for regularization and `Xavire` for network initialization.



### Discussion: performance and convergence speed

To better understand the relationship between the network convergence and activation functions, we demonstrate the loss value and classification error curve for each epoch.

All results are computed based on the median of three runs.

![Twitter Training Loss 2](/resources/Twitter_Training_Loss2.png)
![Twitter Training Error 2](/resources/Twitter_Training_Error2.png)

| Activation functions | Classification Error |
| -------------------- | -------------------- |
| relu                 | 12.248%              |
| prelu                | 12.304%              |
| elu                  | 12.570%              |
| silu                 | 12.416%              |
| mish                 | 12.472%              |
| gelu                 | 12.248%              |

In the first figures, Relu has the fastest converge speed, while Elu is slightly slower than others according to the loss value.

However, as shown in the second figure, the difference in the dropping speed of the loss curve does not reflect the performance. The final classification error shows no difference under different activation function selections.

We suggest that the activation function does not affect the final performance so much if the trained model is fully converged.



### Discussion: robustness to the learning rate

The former section has shown that the performance does not heavily rely on the activation function under a reasonable learning rate. Yet, the acting of the model under a different set of learning rates still remains unclear.

Accordingly, we test the convergence speed and final performance of each activation function under the following settings:

\- Learning rate = [1e-2, 1e-3, 1e-4, 1e-5]

All results are computed based on the median of three runs.



![Twitter Training Loss 1](/resources/Twitter_Training_Loss1.png)
![Twitter Training Error 1](/resources/Twitter_Training_Error1.png)

| Activation functions | Classification Error |
| -------------------- | -------------------- |
| relu                 | 16.765%              |
| prelu                | 15.436%              |
| elu                  | 17.827%              |
| silu                 | 17.030%              |
| mish                 | 17.394%              |
| gelu                 | 16.555%              |

In the figure, there are several oscillations during the convergence process, which implies the unstable nature of the network training under a fast learning rate.

We observed that Elu function has the highest loss value and classification error. Accordingly, we suggest Elu function is the most sensitive to an unstable condition.

Prelu performs slightly better than others, yet it still can only result in sub-optimal performance compared to other settings.



![Twitter Training Loss 2](/resources/Twitter_Training_Loss2.png)
![Twitter Training Error 2](/resources/Twitter_Training_Error2.png)

| Activation functions | Classification Error |
| -------------------- | -------------------- |
| relu                 | 12.248%              |
| prelu                | 12.304%              |
| elu                  | 12.570%              |
| silu                 | 12.416%              |
| mish                 | 12.472%              |
| gelu                 | 12.248%              |



The result trained under lr=1e-3 had been shown in the previous section.

The performance is similar across all activation function settings.

Thus, we selected them as our best model for Twitter-POS dataset.



![Twitter Training Loss 3](/resources/Twitter_Training_Loss3.png)
![Twitter Training Error 3](/resources/Twitter_Training_Error3.png)

| Activation functions | Classification Error |
| -------------------- | -------------------- |
| relu                 | 12.150%              |
| prelu                | 12.276%              |
| elu                  | 13.940%              |
| silu                 | 13.255%              |
| mish                 | 12.933%              |
| gelu                 | 12.458%              |



The curve for `lr=1e-4` is pretty similar to `lr=1e-3`, except the convergence is smoother. On the other hand, the performance is slightly lower but does not show any significant differences.




![Twitter Training Loss 4](/resources/Twitter_Training_Loss4.png)
![Twitter Training Error 4](/resources/Twitter_Training_Error4.png)

| Activation functions | Classification Error |
| -------------------- | -------------------- |
| relu                 | 16.261%              |
| prelu                | 15.674%              |
| elu                  | 15.646%              |
| silu                 | 17.156%              |
| mish                 | 16.443%              |
| gelu                 | 16.695%              |



In the setting of `lr=1e-5`, the loss value and classification error increased compared to the previous settings. We argue that the model does not fully converge due to a slower learning rate.

One interesting finding is that Elu performs the best under such a slow learning rate. We point out that Elu produces a close-to-natural gradient in values closer to zero, which makes it more noise-free than other relu-like functions, especially for small values.



![Twitter Accuracy](/resources/Twitter_Accuracy.png)



Finally, we demonstrate the accuracy of all 6 activation functions under different learning rate settings. In general, `lr1e-3` and `lr1e-4` give us better results compared to the other two settings.

In addition, PRelu shows the best robustness against LR changes as it has the highest performance across different settings.

We attribute it to the adaptive ability of PRelu. While both PRelu and Elu both can propagate negative gradient, only PRelu has the flexibility to adapt the parameters of the rectifiers.





### Discussion: running time

In this section, we compare the training and inference time for the 6 activation functions. Specifically, we run training and inference respectively for 100 epochs and then compute the average spending time for each input sample.

All experiments are conducted on Google Colab.

The hardware specifications are listed as follows:

\- CPU: Intel(R) Xeon(R) CPU @ 2.20GHz

\- Memory size: 13GB

We did not enable any GPU/TPU acceleration.



|       | Avg. training time per samples (μs) | Avg. inference time per samples (μs) |
| ----- | ----------------------------------- | ------------------------------------ |
| relu  | 126.2016                            | 4.5404                               |
| prelu | 128.7802                            | 7.0469                               |
| elu   | 92.9664                             | 5.1761                               |
| silu  | 117.4599                            | 4.8332                               |
| mish  | 133.7703                            | 10.7620                              |
| gelu  | 214.3286                            | 5.5138                               |



![Twitter Training Time](/resources/Twitter_Training_Time.png)
![Twitter Inference Time](/resources/Twitter_Inference_Time.png)



As shown in the figure, Gelu significantly spends more time on computing during training. Computing Gelu involves the estimation cumulative distribution function (CDF) of a Gaussian function. There are several ways to improve the approximation of CDF, yet most of them would bring up extra computation costs.

As for inference time, Mish spends the longest time on computing. We attribute it to the complex nature of the Mish function. Specifically, Mish is a combination of multiple functions, i.e., Tanh and Softplus, which significantly increase its complexity.

For the rest function choices, we did not observe notable differences in efficiency between each of them. 

Note that we only have limit access to Colab server and therefore might not be able to maintain the necessary consistency for each run. Thus, there might be some unfair environmental changes caused by the hardware-/software-side of Colab.
