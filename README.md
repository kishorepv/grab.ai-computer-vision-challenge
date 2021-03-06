### grab.ai-computer-vision-challenge
Classification on Stanford Cars Dataset

This repository contains code to train the Stanford Cars dataset.



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
* One interesting observation was that the model can identify a car well enough just by looking at the front or back of the car. Leveraging this, the train dataset is preprocessed such that each input image is split into halves vertically, each of which is a new image sharing the original label. This effectively doubles the train dataset. This is done by the function `vertical_splitter`. During test time, each image is split using this function and the resultant softmax scores of the two images are averaged and returned as a single prediction.
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

Inference:
* During inference Test Time Augmentation (TTA) is done, which leads to improvement of the model performance. TTA was introduced in [Test-time augmentation with uncertainty estimation for deep learning-based medical image segmentation](https://pdfs.semanticscholar.org/c66a/9706949e7dfb21e7b2304574fb6bd5c3c632.pdf)



Experiments:

Each experiment folder contains a different model which has an associated notebook and test script.
*  `experiment-1`
 * In this experiment, the model is trained on vertically split images (as explained in first point of Feature Engineering) with above mentioned data augmentation. A pretrained `resnet152` architecture is used. The model achieves an accuracy of **88.7%** without TTA and **89.7%** with TTA
 * Trained model can be found in this [Google Drive](https://drive.google.com/drive/folders/1tGeFQ9ZRELc2yfw0t9zjIKJzeMnHuGV8?usp=sharing) under `experiment-1` folder.
 * To evaluate the model on a test set, use `test.py` located in the folder. The results are stored in folder specified by argument `--result_dir`. It contains the `predictions_TIMESTAMP.npy` (raw softmax scores for each test image), `predicted_labels_with_confidence_TIMESTAMP.npy` (confidence scores and the predicted label) and `test_file_names_TIMESTAMP.csv` (test image names to match the predictions).
 * Sample usage of `test.py`
    * `python test.py --test_path=data/cars_test --model_path=data/cars_train_2x --model_name=MODEL-final-stanford-cars-1x2-tfms-res152.pkl --ylabel_path=data/test_labels.csv --result_dir=results` - with TTA
    * `python test.py --test_path=data/cars_test --model_path=data/cars_train_2x --model_name=MODEL-final-stanford-cars-1x2-tfms-res152.pkl --ylabel_path=data/test_labels.csv --result_dir=results --no_tta` - without TTA
* `experiment-2`
 * In this experiment, the model is trained on vertical split images, similar to `experiment-1`. The difference being that the images are pre-processed to extract the car out of the image (using [YOLO detector](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/) in opencv) and these car-crop images are further processed to split each images into two by cutting them vertically in the center. The YOLO detector is pretrained on the PASCAL dataset. This is done by `crop_cars2` function. The code for this is borrowed from [here](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/). This model achieves an accuracy of **90.5%** for both with and without TTA. One problem of this is that the YOLO object-detector in opencv does not run on the GPU and hence is slow.
  * Trained model can be found in this [Google Drive](https://drive.google.com/drive/folders/1tGeFQ9ZRELc2yfw0t9zjIKJzeMnHuGV8?usp=sharing) under `experiment-2` folder.
  * To evaluate the model on a test set, use `test.py` located in the folder. The results are stored in folder specified by argument `--result_dir`. It contains the `predictions_TIMESTAMP.npy` (raw softmax scores for each test image), `predicted_labels_with_confidence_TIMESTAMP.npy` (confidence scores and the predicted label) and `test_file_names_TIMESTAMP.csv` (test image names to match the predictions).
  * Sample usage of `test.py`
     * `python test.py --test_path=data/cars_test --model_path=data/cropped_cars_train_2x --model_name=MODEL-final-stanford-cars-1x2-tfms-yolo-res152.pkl --ylabel_path=data/test_labels.csv --result_dir=results --yolo_dir=data/yolo-object-detection` - with TTA
     * `python eval-final1_2x-yolo-aug-resnet152.py --test_path=data/cars_test --model_path=data/cropped_cars_train_2x --model_name=MODEL-final-stanford-cars-1x2-tfms-yolo-res152.pkl --ylabel_path=data/test_labels.csv --result_dir=results --yolo_dir=data/yolo-object-detection --no_tta` - without TTA
     * You can download the YOLO pretrained weights and config file from [here](https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/opencv-yolo/yolo-object-detection.zip) - source [https://www.pyimagesearch.com](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
* `other-experiments`
 * Contains notebooks for experiments that did not perform as well as `experiment-1` or `experiment-2`
 
Trained models:
* Experiment-wise trained models and other information can be downloaded from [here](https://drive.google.com/drive/folders/1tGeFQ9ZRELc2yfw0t9zjIKJzeMnHuGV8?usp=sharing)
