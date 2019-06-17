### grab.ai-computer-vision-challenge
Classification on Stanford Cars Dataset

This repository contains code to train the Stanford Cars dataset.


Setup:
Requirements:
* `python3`
 * `fastai - 1.0.53.post2` (uses `PyTorch`)

Setup:
* `fastai`
 * `conda install -c pytorch -c fastai fastai`
   * Refer to https://github.com/fastai/fastai for more ways to install

Dataset:
* Train and test data from [Stanford Cars dataset](http://imagenet.stanford.edu/internal/car196/cars_train.tgz)

Model architecture:
* Used a resnet152, pretrained on the Imagenet datasets

Performance:
* Accuracy on the test set of [Stanford Cars dataset](http://imagenet.stanford.edu/internal/car196/cars_train.tgz) is 90%

Feature engineering:
* One interesting observation was that the model can identify a car well enough just by looking at the front or back of the car. So to leverage that, the train dataset is preprocessed such that each input image is split into two vertical halves, each of which is a new image sharing the original label. This effectively doubles the train dataset.
* Most of the images contain non-car information in it. Cropping out the car and using it for prediction can improve the performance of the model. To crop the cars out the images, the [YOLO model](https://docs.opencv.org/master/da/d9d/tutorial_dnn_yolo.html) available in opencv is used

Data Augmentation:
* Random horizontal image flips (probability=0.5)
* With probability of 0.75
 * Random rotation (-10 to 10 degrees)
 * Random zooming (1 to 1.1x)
 * Random lighting and contrast change
 * Random symmetric warp (-0.2 to 0.2)
* Random squishing of image
* Random cutout - as detailed in [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)

Training:
* The model is trained using Cyclic Learning Rate, as enumerated in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)
* For the training, One Cycle Policy which was introduced in [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/pdf/1803.09820.pdf)
* The biggest batch that fits into the GPU is used
* Adam optimizer is used. Weight decay for Adam is handled as detailed in [Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101.pdf)
* The model is trained for four epochs with all the layers frozen, except the last FC layer. After that the whole model is trained for 15 more epochs.
* While training the while network discriminative/differential learning rate is used. In this way the earlier layer have a slower learning rate than the later layers.

