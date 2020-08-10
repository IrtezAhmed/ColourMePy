# ColourMePy
Image Colourization with Convolutional Autoencoder

# 1.0 Introduction
Our project is image colorization, the process of colourizing grayscale images. As seen in Figure 1, inputting a grayscale image into the AI model will return a colored version of the input image. The AI model will be trained to give the correct colors, in the LAB color space, to the correct objects. For example, the features of grass would be recognized by the model, and would return the color green. A practical use for this could be to turn old greyscale photos into colored ones. This can also have a more modern use as it can turn grayscale mangas, tv-shows and movies into colored versions of it, which could let the viewer have a new way of enjoying the art. Machine learning is a reasonable approach for accomplishing these feats as it could accurately, easily, and quickly take an input greyscale picture and modify it into a colored one with minimal supervision.

# 2.0 Background and Related Works
Research on image colorization algorithms have demonstrated that models can be trained to produce reasonably colorized images from grayscale, and our model attempts to replicate the results of this research. A major study done by Richard Zhang et al. at the University of California, Berkeley details the use of GANs to train a fully-automated model that colorizes grayscale photographs [1, 2] The goal of Zhang’s model was to create plausible colourized images such that they could deceive a human into believing that they are original. The model emphasizes rare colors and strives for diversity in the colour palette to achieve a believable output. However, they found that biases in training datasets cause many algorithms to neglect the multimodal nature of colorization, in that certain images can have a range of colors (eg. an apple can either be red, green, or yellow). Our model does not attempt to diversify our dataset because we are training a low number of images compared to Zhang, who trains about three millions images on their model.
3.0 Data Processing
Our data was taken from Shravankumar Shetty’s Image Colorization dataset hosted on Kaggle [5]. The dataset contains four .npy files (numpy arrays) split as follows:

gray_scale.npy
25000 grayscale images
ab1.npy
first 10000 coloured images
ab2.npy
next 10000 coloured images
ab3.npy
final 5000 coloured images

The images contained in these files are 224x224px and exist in the LAB colour space. First, the grayscale images had to be combined with coloured images because the numpy arrays do not show a fully colourized image on their own.

The function not only combines the input images, but it also returns an array with a specified batch size to allow our team to choose the amount of training and validation data in their respective sets. Choosing smaller amounts was also crucial to minimize the RAM used, because Colab would crash for batch sizes in the thousands. 
Visualizing sample images in the LAB colour space proved difficult for our team. So we opted to create a function that converts sample images into the RGB colour space just for our previews. The function and an arbitrary preview is as follows in Figure 3 and 4.

Figure 3: Code for previewing a random image

Figure 4: Code to test if the data has been loaded
3.1 Splitting the Dataset 
Initially, we planned to split the dataset into 20000 for training, the remaining 5000 for validation, and then an extra 3000 images obtained by our team for testing. After learning that our primary model in Section 4 has long training times that make hyperparameter tuning inefficient (~3 hours), we decided to change how we split our data. Now we split our data in 10000 images for training, 10000 images for validation, and the remaining 5000 for testing. This meant that we did not need to search for additional images outside of those provided in the Kaggle dataset.
4.0 Primary Architecture
For our primary architecture, we chose to use a convolutional autoencoder. The encoder has two convolutional layers, whereas the decoder has three convolution layers, each followed by ReLu activation to remove negative values that do not fit in the standard LAB colour range. While training, we used mean square error for our loss and Adam for our optimizer. After much testing, we decided to leave the learning rate and weight decay to the Adam optimizer which provided us with the best results.

Unlike the baseline model in Section 5.0, this model takes three input channels. The 224x224 L colour channels in LAB were appended to two zero arrays of 224x224 representing A and B, thus representing all 3 channels in the input. The model upsamples the data from three channels to 64 channels, then back down to 3 channels. This process populates the zero arrays representing A and B, providing a fully colourized image. After trial and error, we found that the architecture shown in Figure 5 has provided the best results, with an accuracy of 81.9% (See Section 6). A padding of 1 was added to the encoding convolutional layers to yield the correct output size. 


# 5.0 Baseline Model
As can be seen in Figure 6, a simple ANN with three fully connected linear layers was chosen for our baseline model. The layers upsample the input by one each layer until it reaches 3 channels representing the LAB colour space, which can be seen in Table 3. This model has an accuracy of 61.7% (See Section 6). The images of the baseline produced can be seen below in Figure 7.

It was immediately clear after testing this model that it was biased towards the colours blue and orange. Moreover, the images themselves appear to have a filter that makes them appear artificial with excess hue and saturation. 

6.0 Quantitative Results
For our convolutional autoencoder, we used mean square error loss to find the accuracy and loss of our model, where N is the number of pixels in the image (224x224 = 50176 pixels):

loss = 1N(Prediction - Ground Truth)2

As in the equation above, we compared each pixel of our model’s prediction to its ground truth counterpart and then found the average over each pixel. The highest possible loss using this formula is 65025, whereas 0 is the lowest. By using this measurement we managed to get this down to ~6.0 after training the model. However using this as a metric to determine accuracy was not representative of our model’s performance because any value greater than 7500 would not be visually comprehendible and values under 7500 would not be clear enough to consider a colourization. Thus, 7500 was considered the maximum loss for calculating accuracy, where:

accuracy = (max.loss - lossmax. loss)100%

Since our baseline model had an average loss of 2873, our accuracy came out to be 61.7%. Our primary model has an average loss of 1358 over the 10000 training images. Thus, the accuracy came out to be 81.8%. An explanation as to why our results were not better can be found in Section 7. The following table summarizes the results of our primary and baseline models: 

Model
Accuracy
Training Time
Baseline (Linear ANN)
61.7%
~30 minutes
Primary (Convolutional Autoencoder)
81.9%
~150 minutes

Table 4: Summary of trained models
7.0 Qualitative Results

As aforementioned, our model is supposed to take a grayscale input image and convert it into a fully colored image. Below in Figure 8 and 9, you can see samples of our colourization predictions from our primary model.


Figure 8: On the left, the inputted image, and on the right is the output image of our model.

Figure 9: On the left, the inputted image, and on the right is the output image of our model.

While the images could potentially fool a human being into believing it is an original colourization, some images tend to have some arbitrary hues. Moreover, somes images also tend to contain irregular particles, which makes the image seem rather unnatural (Figure 10). These are likely a result of convolutions.


Figure 10: Input and output in our model of a hazelnut, depicting odd particles.

The output image from the model may also have assigned wrong colors to certain objects due to the object having many different potential colors. In Figure 11, we input an image of a flower into our model. Due to a flower having many different potential colors, the output flower gets assigned the wrong color, orange,  compared to the ground truth, which is blue.


Figure 11: Input (Left), Prediction (Middle), and Ground Truth (Right)
8.0 Evaluate Model on New Data
As our team did not expose our model to 5000 images from our dataset, these images were available for testing without having any influence over our model’s hyperparameters. As such, we are able to also show the corresponding ground truth for each prediction. The following table has six predictions our model made on unseen data.






Table 5: Results of our model on new data

Note, that ground truth for Image 4 seemed to accidentally be a copy of the grayscale image in the dataset. We included this image to show that the ground truth had no effect on the accuracy of our colourizations here. While these trials manage to capture the essence of the ground truth in its colourization, the model cannot correctly predict what colour certain things should be such as the skyline in Image 3 or the windows in Image 2. In addition to the reasons mentioned in Section 7, the number of training samples we provided were minimal compared to typical amounts which are in the millions. We also mention additional reasons for our model’s poor accuracy in Section 9, including the use of GANs and training time. 
9.0 Discussion
9.1 Interpretation of Results
At an accuracy of 81.9%, our model’s performance was sufficient enough to recreate a relatively accurate depiction of our grayscale images. If we used our model to colour old movies or historical photos, we could definitely get an idea of what the scene looked like in colour. However, after 2.5 long hours of training, it was far from ideal. Compared to our ANN, the extra 20% of accuracy in our main model is worth it, but at a training time five times longer, our model did not prove to be efficient. Furthermore, our qualitative results showed that our model would add a brownish hue to the photos that resembles the “sepia” filter and oddly shaped particles that we believe could be due to undertraining. Though this could prove to have interesting artistic implications and our model could be used as a distortion filter for social media purposes, for the purpose of our project it made the results less accurate. There are a number of reasons as to why our model’s performance could not be increased further. 

9.2 Performance of Model
The first issue was that colour is subjective and our model lacked the ability to correctly discern correct colourizations. Our model was successful in colourizing simple features, like the sky as blue or grass as green. However, our model struggled to accurately colour features like flowers or people due to this subjectivity. The only way to overcome this is to increase the number of images and the diversity of the dataset; however, even then we would not be able to replicate human instinct. 

Some other surprises that we faced when designing our model was that data cleaning was much more complicated than expected. Since we were working in the LAB colour space, we had to manipulate the data differently than if it were in the RGB color space. Having to work with unfamiliar data and tackle a new type of problem caused us to perform much more trial and error than anticipated. Most of the problems we have dealt with previously were simple classification problems, so having to colourize an image proved to be an interesting challenge.

9.3 Improvements
An substantial improvement that could be made to our model is the use of Generative Adversarial Networks (GAN). GANs learn the structure of the input data to generate new data. The model would be trained so that the generator produces more realistic colourization while the discriminator becomes better at distinguishing it from the ground truth. The goal would be to train the model until the generator produces a colourization that the discriminator cannot differentiate from the ground truth. As seen in research studies, GANs are capable of producing better colourizations; however, it would be more difficult to train, since we would have to train two networks. 

9.4 Project Difficulty
The task of image colourization is fundamentally a regression problem. The complexity of these problems lies in that there is no definitive value that would result in a good prediction. Unlike classification problems wherein the model only has to make a prediction out of a small and finite set of outcomes, an image colourization problem necessarily requires that the model correctly predicts 150528 different values within appropriate ranges between 0 and 255 in the LAB colour space (224x224x3=150528). Due to the multimodal nature of coloured images, a grayscale image can adopt numerous correct colourings. It is far from a simple classification problem and it is next to impossible to achieve perfect colourization when even humans cannot determine the correct colourization of certain images on their own. Despite the immense difficulty of this problem, our model was able to create a realistic colourization of our grayscale images, thus achieving our main goal. 

10.0 Ethical Considerations
Image colourization models are only effective on images similar to the dataset over which it is trained. A model trained on a biased data set may miscolor certain images and potentially offend people’s belief and identities. This would include training a model that miscolours a tan-skinned male as white or miscoloring a religious figure like Vishnu because of a biased dataset. Additionally, the conversion of images from grayscale to colorized may be considered tampering with images without the consent of the owner. While we did not create our dataset, we also did not train over a sufficient number of images to counter biases. Our model is not capable of colouring some non-controversial images in the first place, and so balancing our dataset would be a consideration for a future project that expands on this one.
