# ColourMePy
Image Colourization with Convolutional Autoencoder

## 1.0 Introduction

(Note, to duplicate our results, any files that are run should be run in Google Colab with the appropriate paths to the dataset. Else, the appropriate modules should be imported for local use).

This project is image colorization, the process of colourizing grayscale images. This project was completed as a part of a machine learning course at the University of Toronto

As seen in Figure 1, inputting a grayscale image into the AI model will return a colored version of the input image. The AI model will be trained to give the correct colors in the LAB color space, to the correct objects. For example, the features of grass would be recognized by the model, and would return the color green. A practical use for this could be to turn old greyscale photos into colored ones. This can also have a more modern use as it can turn grayscale mangas, tv-shows and movies into colored versions of it, which could let the viewer have a new way of enjoying the art.

## 2.0 Background
Research on image colorization algorithms have demonstrated that models can be trained to produce reasonably colorized images from grayscale, and our model attempts to replicate the results of this research. However, while most research tends towards GANs, our project uses an autoencoder for simplicity's sake. A major study done by Richard Zhang et al. at the University of California, Berkeley details the use of GANs to train a fully-automated model that colorizes grayscale photographs. The goal of Zhang’s model was to create plausible colourized images such that they could deceive a human into believing that they are original. Their model emphasizes rare colors and strives for diversity in the colour palette to achieve a believable output. However, they found that biases in training datasets cause many algorithms to neglect the multimodal nature of colorization, in that certain images can have a range of colors (eg. an apple can either be red, green, or yellow). Our model does not attempt to diversify our dataset because we are training a low number of images compared to Zhang, who trains about three millions images on their model.

## 3.0 Data Processing
The data was taken from Shravankumar Shetty’s Image Colorization dataset hosted on Kaggle. Check out his dataset here: https://www.kaggle.com/shravankumar9892/image-colorization  The dataset contains four .npy files (numpy arrays) split as follows:


gray_scale.npy | ab1.npy | ab2.npy | ab3.npy
--- | --- | --- | --- 
25000 grayscale images | first 10000 coloured images | next 10000 coloured images | final 5000 coloured images

The images contained in these files are 224x224px and exist in the LAB colour space. First, the grayscale images had to be combined with coloured images because the numpy arrays do not show a fully colourized image on their own.

## 4.0 Primary Architecture
For our primary architecture, we chose to use a convolutional autoencoder. The encoder has two convolutional layers, whereas the decoder has three convolution layers, each followed by ReLu activation to remove negative values that do not fit in the standard LAB colour range. While training, we used mean square error for our loss and Adam for our optimizer. After much testing, we decided to leave the learning rate and weight decay to the Adam optimizer which provided us with the best results.

Unlike the baseline model, this model takes three input channels. The 224x224 L colour channels in LAB were appended to two zero arrays of 224x224 representing A and B, thus representing all 3 channels in the input. The model upsamples the data from three channels to 64 channels, then back down to 3 channels. This process populates the zero arrays representing A and B, providing a fully colourized image. After trial and error, we found that this architecture has an accuracy of 81.9% (See Section 6). A padding of 1 was added to the encoding convolutional layers to yield the correct output size. 

## 5.0 Quantitative Results
For our convolutional autoencoder, we used mean square error loss to find the accuracy and loss of our model, where N is the number of pixels in the image (224x224 = 50176 pixels):

![MSE Loss](https://user-images.githubusercontent.com/61393006/90329818-d707c600-df75-11ea-95dd-284cfd83c30e.png)

As in the equation above, we compared each pixel of our model’s prediction to its ground truth counterpart and then found the average over each pixel. The highest possible loss using this formula is 65025, whereas 0 is the lowest. By using this measurement we managed to get this down to ~6.0 after training the model. However using this as a metric to determine accuracy was not representative of our model’s performance because any value greater than 7500 would not be visually comprehendible and values under 7500 would not be clear enough to consider a colourization. Thus, 7500 was considered the maximum loss for calculating accuracy, where:

![Accuracy Calculation](https://user-images.githubusercontent.com/61393006/90329839-f7378500-df75-11ea-949c-29749d576bfe.png)

Since our baseline model had an average loss of 2873, our accuracy came out to be 61.7%. Our primary model has an average loss of 1358 over the 10000 training images. Thus, the accuracy came out to be 81.8%. An explanation as to why our results were not better can be found in Section 7. The following table summarizes the results of our primary and baseline models: 

Model | Accuracy | Training Time
--- | --- | ---
Baseline (Linear ANN) | 61.7% | ~30 minutes
Primary (Convolutional Autoencoder) | 81.9% | ~150 minutes


## 6.0 Qualitative Results

Below are some samples of our model's performance (left is grayscale, while right is model prediction).

Here's how our model performed on this image of a ship:

![boat](https://user-images.githubusercontent.com/61393006/90329910-a4aa9880-df76-11ea-86c4-083125b7a37c.png)

Here is how our model performed on an image of a mountain:

![mountain](https://user-images.githubusercontent.com/61393006/90329911-a70cf280-df76-11ea-98c3-1084519290da.png)

While the images could potentially fool a human being into believing it is an original colourization, some images tend to have some arbitrary hues. Moreover, somes images also tend to contain irregular particles, which makes the image seem rather unnatural:

![walnuts](https://user-images.githubusercontent.com/61393006/90329936-f18e6f00-df76-11ea-8530-d358298c35ac.png)

The output image from the model may also have assigned wrong colors to certain objects due to the object having many different potential colors. We tried to input an image of a flower into our model. Due to a flower having many different potential colors, the output flower gets assigned the wrong color, orange,  compared to the ground truth, which is blue.

![subjective colours](https://user-images.githubusercontent.com/61393006/90329943-066b0280-df77-11ea-83ba-3d88159ea232.png)

## 7.0 More results!

![A list of results](https://user-images.githubusercontent.com/61393006/90329976-5ea20480-df77-11ea-8363-abc9de63523b.png)

![More results](https://user-images.githubusercontent.com/61393006/90329983-6bbef380-df77-11ea-983e-a5e22072b105.png)

More models to come!
