# COVID-19 Classification Through Various Methods 

Part 1:
<a href="https://colab.research.google.com/github/Lewis34cs/corona_audio/blob/main/covid_proj_p1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Part 2:
<a href="https://colab.research.google.com/github/Lewis34cs/corona_audio/blob/main/covid_proj_p2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Author**: Christopher Lewis

## Description

The contents of this repository describe methods of COVID-19 Classification through multiple testing methods. The analysis displayed below is detailed in hopes of making the work accessible and able to be replicated by others wishing to explore and analyze this project themselves.

From the beginning of 2020, COVID-19 has run rampant throughout the entirety of the globe, resulting in over 100 million cases and 2 million deaths. While vaccines are beginning to be pushed out as our new line of defense, my project's goal is to identify other ways to detect COVID-19 in a patient to help diagnose those who are infected and slow the spread. In this project, I focus on two different ways to identify COVID-19: multi classification via chest x-rays and binary classification through coughing audio. Both will be using Sequential Convolutional Neural Networks, and the code for this project can be found in the google colab link at the top of the page. To run the Google Colab code, you must allow Google Colab to access an account. I provide links for the data in the Google Colab Notebook (link above). Please be sure to save zipped folders for these audio datasets into your google drive so you can access them on Google Colab. The chest x-ray database was obtained via Kaggle's API.

In this project we are going to be using the **OSEMN Process**:

**Obtain**: Our datasets came from a variety of places including: Kaggle via API, Stanford Unversity's Virufy App via Github, and the CoughVid app via Zenodo's website.

**Inspecting, Scrubbing, & Exploring**: 

PART 1: When dealing with the chest x-rays, we will be making sure that our train, test, and validation sets are stocked with an appropriate amount of images, examine class proportions, and use CLAHE as our key preprocessing technique. 

PART 2: We will be implementing scrubbing techniques on the CoughVid dataset according to a certain thresholds and custom filters. Other preprocessing and preparation techniques include converting .webm video files to .ogg files, setting the length of our audio files to a desired length through zero padding shorter files, and creating mel-spectrogram images via the librosa library.

**Modeling & Exploring**:

PART 1: We will then create Sequential models to train on our unpreprocessed and CLAHE preprocessed images to see if implementing CLAHE as a preprocessing technique proves to be effective.

PART 2: We will create and train Sequential models for this section as well to try and find if our models are able to correctly classify the right class for each image. We will also save augmented spectrogram images of the minority class in our training set in an attempt to help the model identify the proper class.

**Interpretion**: Here we will give our results of our models, conclusion, recommendations, and further research.

# PART 1 - Chest X-ray Multi Classification

In Part 1, we explore a database retrieved from Kaggle's API containing 3800+ high quality images that were diagnosed in a professional setting to three classes: COVID, Healthy, Viral Pneumonia. We create an evaluate a baseline model, evaluate a basic CNN model, and then proceed to use a preprocessing technique called CLAHE to generate new images, which a new model is then trained on. We evaluate that model's results and discuss further thoughts.

## Obtaining COVID CXR dataset via Kaggle API

We start by importing the chest-ray dataset by making a call to the Kaggle API. In order to make a call to the API, you must request a key from Kaggle. I've saved my key to my local storage, so I use the upload() function to access my key from my local files. Once we've uploaded our API key, we make a new hidden folder to store our key in, and make sure that only we can access and view this key. After the dataset is downloaded into our current working directory, we unzip the file and extract all files within the folder to the root directory. 

Note that if you would like to choose a different place to extract the folder contents, in the zipf.extractall() function, simply type the address of where you would like the contents to be saved. Since the run time for this portion of the project is relatively fast, we save it to the root directory. Note that saving it to the root directory while in Google Colab makes it a temporary folder - so if you disconnect the runtime, you will have to feed in your key and extract the files again.

#### Identifying the number of images in our dataset

From here, we view the extracted contents and count the number of images per class along with the total image count. We see that we are working with 3800+ images. We can also see that our classes are relatively balanced.

### Splitting our images into train, test, and validation sets

The splitfolders library proved to be extremely effective and very easy to use. It allowed us to feed in a directory address (base_folder variable) and gave it a place to put the new train, test, and validation folders (output variable). Note that the output address must already be existing. We also set a seed in order to help with model score reproducibility when training our models. The ratio parameter required floats that determined the percentage of data going to each folder. 

If you would like more information about splitfolders library, please visit https://github.com/jfilter/split-folders

#### Recommendation:

If you prefer working with folder structures when dealing with training and testing data, I would highly recommend using the splitfolders library. Through the use of this library, I was able to create train, test, and validation folders, create image data generators, and have those generators access the data by flowing from my directories. 

### Baseline Model

We trained a baseline model that received a 34% accuracy score on classifying chest x-rays. In order to use the DummyClassifier, we had to manipulate image data generators to create train, test and validation sets by specifying a batch size equal to total amount of images for each set(80% of images in train, 20% in test, and 10% in validation). The Dummy Classifier tended to classify the majority of x-rays as COVID. 


```python
#Printing classification report and plotting confusion matrix
print(metrics.classification_report(y_test_ohe, y_pred));

plt.figure(figsize=(7, 6))
cm = metrics.confusion_matrix(y_test_ohe, y_pred, labels = [0, 1, 2], 
                              normalize='true')
sns.heatmap(cm, cmap="Greens", annot=True, square=True)
plt.show()
```

                  precision    recall  f1-score   support
    
               0       0.32      0.64      0.42       240
               1       0.36      0.26      0.30       269
               2       0.38      0.14      0.21       270
    
        accuracy                           0.34       779
       macro avg       0.35      0.35      0.31       779
    weighted avg       0.35      0.34      0.31       779
    
    


![png](/images/output_18_1.png)


From here, we created new image data generators using the chest x-ray images, setting the batch size to 32 and fed the generators into a CNN model. The model received a 94% accuracy score, approximately 180% better than our baseline model. 

## Preprocessing Techniques

### CLAHE Information

Contrast Limited Adaptive Histogram Equalization (CLAHE) is used to equalize pixel intensity in images. It is very similar to Adaptive Histogram Equalization (AHE), except it doesn't over-amplify the contrast of the image. This is controlled by the clipLimit parameter. The way CLAHE works on an image is that it focuses on small portions of the image (tileGridSize parameter) and then combines these portions together through bilinear interpolation to help remove any artificial boundaries, which means that it enhances the local contrast of the total image. This essentially helps with the pixel intensity distribution, allowing us to see more "depth" in an image.

link for info on cv2.createCLAHE():
 - https://docs.opencv.org/master/d6/dc7/group__imgproc__hist.html#gad689d2607b7b3889453804f414ab1018

Through the use of CLAHE, we can definitely see more of the infiltrate in the lung areas of the chest x-ray. These infiltrate areas of the lung can determine whether or not a person has Pneumonia. According to Hosseiny et al., when there is radiographic appearances of multifocal ground glass opacity, linear opacities, and consolidation, these are usually seen in cases of coronavirus type infections, including COVID-19, SARS, and MERS.

Now that we've preprocessed our CXR images and split them into train, test, and validation folders, we create our generators, define class weights and model, and train the model on the preprocessed images. We will be using the same model structure before to measure the effectiveness of CLAHE by comparing the recall rate of the COVID class and the overall accuracy of the two models.


```python
best_model = load_model('.../cxr_models/model-05-0.101.hdf5')
class_report_gen(best_model, cl_test_generator, 
                 class_indices=test_generator.class_indices, cmap='Greens')
best_model.evaluate(cl_test_generator, verbose=1)
```

    ---------------------------------------------------------
                      Classification Report
    
                  precision    recall  f1-score   support
    
               0       0.98      0.98      0.98       240
               1       0.95      0.98      0.96       269
               2       0.98      0.94      0.96       270
    
        accuracy                           0.97       779
       macro avg       0.97      0.97      0.97       779
    weighted avg       0.97      0.97      0.97       779
    
    ---------------------------------------------------------
    


![png](/images/output_26_1.png)


    25/25 [==============================] - 7s 271ms/step - loss: 0.1188 - acc: 0.9679 - precision: 0.9704 - recall: 0.9679 - auc: 0.9941
    




    [0.11884351074695587,
     0.9679075479507446,
     0.9703989624977112,
     0.9679075479507446,
     0.994138777256012]



By using our CLAHE CXR images, we are able to improve our model's acuracy to around 97%, which is 3% better than our model that used the unpreprocessed CXR images. We should also note that this model's recall score for classifying the COVID class is 98%, with only a 2% false negative rate. Using computer vision is a great way to diagnose certain infections and diseases through the use of chest x-rays.

#### Recommendation: 
When working with images like x-rays or MRI scans, I highly recommend using CLAHE as a preprocessing technique to create new images that give the model more to learn from. CLAHE also is able to provide enough contrast to the image without overamplifying the intensity of the pixels. It is a great tool if the goal of your project involves detection and/or recognition with images.

#### Lime Image Explainer

We create a function that implements the library called lime, which allows us to see what the model determines as most important when classifying an image. Below we can see in green what the model identifies as positive correlations with the COVID class, and red as negative correlation with the class.


```python
explain_image(basic_prep_cnn, cl_train_generator, num_samples=2000, num_feats=3, 
              class_label=0)
```


    HBox(children=(FloatProgress(value=0.0, max=2000.0), HTML(value='')))


    
    


![png](/images/output_31_2.png)


## Interpretation & Further Thoughts

While using x-rays to diagnose patients with COVID has proven to be successful with the model we've created, we should consider the cost and risk of a COVID patient getting a chest x-ray. It would not be ideal for someone with COVID to come into a medical facility and expose other people to the virus. Not only would they be exposing COVID-19 to the medical staff, but also to someone with a potentially lowered immune system or someone who could be at a greater risk of hospitalization if they were to get COVID-19. We should also consider the price of getting an x-ray. A person who does not have health insurance can spend, on average, around $370.00 for a chest x-ray. Furthermore, those who are asymptomatic would not think to get an x-ray if they are not displaying any symptoms.  

# PART 2

In Part 2, we explore the possiblities of using cough audio from healthy and COVID infected individuals and see if we can create a model that can accurately diagnose those with COVID. Using datasets obtained from Stanford's Virufy app and the CoughVid app, we combine these audio files together and create spectrograms off of each audio file. Then we train a model on the created spectrogram images. The end goal is creating a model that can classify our spectrogram images with high degree of accuracy and recall for the COVID class, then proceeding to build an application around the model that people would be able to interact with. Ideally, the application would gather audio input from people who allow the application to record their voice while they cough into the microphone. From there, the program would create a spectrogram image of the inputted audio file and then the model would evaluate the spectrogram and attempt to classify if the audio was COVID positive or not. This would be a free app and would be accessible to everyone with working phone or computer. This would also allow for people to be tested on a daily basis in quick succession when compared to other current testing methods such as Viral Testing or Antibody testing.

## Obtaining Virufy audio data

The Virufy data came from the University of Stanford. While is does not contain many samples, the 16 patients in this dataset have been laboratory-confirmed cases as either having COVID-19 or being healthy at the time their audio was recorded. We will be focusing on the segmented audio in our project, which gives us approximately 121 audio files to train a model. To access the data through Google Drive, we must unzip the folder that is currently stored in our drive.

We see that the virufy audio data we have is all the same length (approx. 1.6 secs in length). Spectrograms take into account the dimension of time, so having our audio files the same length is important. Think of having different time lengths the same as distorting images by either stretching our shrinking them in width. In order for our model to perform the best, we must make sure that our images dimensions are the same.

### Working with Audio Using the Librosa Library

Visualizing our audio example's waveform using librosa.display (ldp)



```python
plt.figure(figsize=(12, 5))
ldp.waveplot(signal, sr=sr)
plt.title('Waveplot')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
```


![png](/images/output_41_0.png)


Visualizing the Fast Fourier Transform


```python
#Fast Fourier Transformation
fft = np.fft.fft(signal)

# These magnitudes indicate the contribution of each frequency in the sound
magnitude = np.abs(fft)

# mapping the magnitude to the relative frequency bins using np.linspace()
frequency = np.linspace(0, sr, len(magnitude))

# We only need the first half of the magnitude and frequency to visualize the FFT
left_mag = magnitude[:int(len(magnitude)/2)]
left_freq = frequency[:int(len(frequency)/2)]

plt.plot(left_freq, left_mag)
plt.title('Fast Fourier Transform')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()
```


![png](/images/output_43_0.png)


Here we have a Fast Fourier Transform plotted. The magnitudes indicate the contribution of each frequency in the sound. The larger the magnitude, the heavier the contribution of the frequency. Here we can see that the majority of the energy resides in the lower frequencies. The only issue with the FFT is the fact that it is static; there is no time associated with this plot. So in order to incorporate time into our audio to see what frequencies impact at what time, we should use the Short-Time Fourier Transformation.

Visualizing a Spectrogram in amplitude


```python
# number of samples per fft
# this is the number of samples in a window per fast fourier transform
n_fft = 2048

# The amount we are shifting each fourier transform (to the right)
hop_length = 512

#Trying out Short-time Fourier Transformation on our audio data
audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

# gathering the absolute values for all values in our audio_stft variable
spectrogram = np.abs(audio_stft)

# Plotting the short-time Fourier Transformation
plt.figure(figsize=(8, 7))
ldp.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz', hop_length=hop_length)
plt.colorbar(label='Amplitude')
plt.title('Spectrogram (amp)')
plt.show()
```


![png](/images/output_46_0.png)


As we can see (or not see), most of the frequencies in our audio contribute very little amplitude to the overall sound. 
Because what we are looking for is not linear, like loudness, we are going to take the log of our sounds amplitude and turn it into decibels. Humans experience frequency logarithmically, not linearly.


```python
to_spectrogram(signal, sr=48000, hop_length=128, n_fft=1024, vmin=-40, vmax=30)
```


![png](/images/output_48_0.png)


A spectrogram is basically composed of multiple Fast Fourier Transforms, where each FFT is calculated on overlapping windowed portions of the signal. In order for us to visualize “loudness” in our signal, we must convert from amplitude to decibels.This allows us to view the loudness of frequencies over time. By switching from a scale in amplitude to decibels, we create an image with more information to give to our model.

#### Recommendation:

When working with audio data, I highly recommend using the librosa library. It is full of different functions that are easy to use and come with great explanations on how to use them. Librosa also focuses on being user friendly and it relies on numpy datatypes, which allows interoperability from librosa to another library. Furthermore, librosa has other functions that allow us to extract different features from an audio file, which could also be used to classify an audio file.


```python
# Creating a mel-spectrogram
to_mel_spectro(signal, sr, hop_length=128, n_fft=1024, figsize=(10,8), vmin=-40, 
               vmax=20, ref=1, n_mels=128)
```


![png](/images/output_51_0.png)


A mel-spectrogram is a spectrogram where the frequencies are converted to the mel-scale. According to the University of California, the mel-scale is “a perceptual scale of pitches judged by listeners to be equal in distance from one another”. We can picture this as notes on a musical scale:

From C to D is one whole step, and from D to E is another whole step. Perceptually to the human ears, the step sizes are equal. However if were were to compare these steps in hertz, they would not be equal steps. A C is around 261.63 Hz, a D is 293.66 Hz, and an E is 329.63 Hz. 

 - C to D difference = 32.03 Hz
 - D to E difference = 35.37 Hz

As the notes go higher in octave, the difference between the steps dramatically increases. Mel-spectrograms provide a perceptually relevant amplitude and frequency representation.


#### Recommendation:

When working with spectrograms created from human audio, taking the log of the amplitude and converting it to decibels will give your model more to look at, and allow it to learn more from each image. Since we are working with coughing audio, converting the frequency to the mel-scale allows us to peer more into the tonal relationship of the frequencies for each audio file. 

### Creating & Saving Mel-Spectrograms for Virufy Dataset

In order for us to train a Sequential Convolutional Neural Network on mel-spectrograms, we create and save mel-spectrograms for each audio clip in our Virufy dataset and save them into a new folder under their respective class. 


```python
# Using np.hstack() to show our images side by side
res = np.hstack((neg_img, pos_img))
# Creating a figure and adding a subplot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
# plotting the horizontal stack using plt.imshow()
plt.imshow(res, cmap='gray')
plt.title('Healthy Image                                 COVID Image')

# Hiding our axes and frame
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
plt.show()
```


![png](/images/output_56_0.png)


When looking at the spectrograms, there's not much we can take away just by looking at them. We notice the coughs are roughly the same length in duration, and the prominence of some frequencies of sound may be slightly different when comparing patients (male vs female, older vs younger, etc). 

### Modeling off of the virufy spectrogram images


```python
viru_hist1 = fit_plot_report_gen(viru_model1, viru_train_gen, viru_test_gen, 
                                 viru_val_gen, epochs=25, 
                                 class_weights=class_weights_dict)
```

    Epoch 1/25
    3/3 [==============================] - 15s 6s/step - loss: 0.9946 - acc: 0.4283 - precision: 0.3885 - recall: 0.8030 - auc: 0.5128 - val_loss: 0.6417 - val_acc: 0.6364 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - val_auc: 0.7143
    Epoch 2/25
    3/3 [==============================] - 12s 4s/step - loss: 0.7511 - acc: 0.5019 - precision: 0.4164 - recall: 0.6020 - auc: 0.4828 - val_loss: 0.6880 - val_acc: 0.3636 - val_precision: 0.3636 - val_recall: 1.0000 - val_auc: 0.8929
    Epoch 3/25
    3/3 [==============================] - 12s 4s/step - loss: 0.6776 - acc: 0.4743 - precision: 0.4112 - recall: 0.8229 - auc: 0.6353 - val_loss: 0.6667 - val_acc: 0.8182 - val_precision: 0.7500 - val_recall: 0.7500 - val_auc: 0.8214
    Epoch 4/25
    3/3 [==============================] - 12s 4s/step - loss: 0.6879 - acc: 0.5158 - precision: 0.4108 - recall: 0.5614 - auc: 0.5437 - val_loss: 0.6676 - val_acc: 0.8182 - val_precision: 0.7500 - val_recall: 0.7500 - val_auc: 0.8750
    Epoch 5/25
    3/3 [==============================] - 12s 4s/step - loss: 0.6868 - acc: 0.5497 - precision: 0.4128 - recall: 0.4424 - auc: 0.5105 - val_loss: 0.6612 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.8571
    Epoch 6/25
    3/3 [==============================] - 12s 4s/step - loss: 0.6631 - acc: 0.6948 - precision: 0.6168 - recall: 0.6193 - auc: 0.7522 - val_loss: 0.6310 - val_acc: 0.7273 - val_precision: 0.5714 - val_recall: 1.0000 - val_auc: 0.8571
    Epoch 7/25
    3/3 [==============================] - 12s 5s/step - loss: 0.6053 - acc: 0.7370 - precision: 0.6420 - recall: 0.6739 - auc: 0.8312 - val_loss: 0.5650 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.8571
    Epoch 8/25
    3/3 [==============================] - 12s 4s/step - loss: 0.5720 - acc: 0.7091 - precision: 0.5938 - recall: 0.7280 - auc: 0.7875 - val_loss: 0.5256 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.9286
    Epoch 9/25
    3/3 [==============================] - 12s 4s/step - loss: 0.4473 - acc: 0.8287 - precision: 0.7519 - recall: 0.8979 - auc: 0.8945 - val_loss: 0.5003 - val_acc: 0.7273 - val_precision: 1.0000 - val_recall: 0.2500 - val_auc: 0.9643
    Epoch 10/25
    3/3 [==============================] - 12s 5s/step - loss: 0.5404 - acc: 0.7630 - precision: 0.9130 - recall: 0.4274 - auc: 0.8880 - val_loss: 0.9876 - val_acc: 0.6364 - val_precision: 0.5000 - val_recall: 1.0000 - val_auc: 0.9107
    Epoch 11/25
    3/3 [==============================] - 12s 5s/step - loss: 0.4634 - acc: 0.7415 - precision: 0.6503 - recall: 0.9270 - auc: 0.9136 - val_loss: 0.4061 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.9643
    Epoch 12/25
    3/3 [==============================] - 12s 5s/step - loss: 0.3455 - acc: 0.9009 - precision: 1.0000 - recall: 0.7458 - auc: 0.9551 - val_loss: 0.3471 - val_acc: 0.9091 - val_precision: 1.0000 - val_recall: 0.7500 - val_auc: 0.9643
    Epoch 13/25
    3/3 [==============================] - 12s 4s/step - loss: 0.3795 - acc: 0.9049 - precision: 0.9815 - recall: 0.7564 - auc: 0.9596 - val_loss: 0.5092 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.9821
    Epoch 14/25
    3/3 [==============================] - 12s 4s/step - loss: 0.2384 - acc: 0.9271 - precision: 0.8481 - recall: 0.9848 - auc: 0.9949 - val_loss: 0.3514 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.9643
    Epoch 15/25
    3/3 [==============================] - 12s 5s/step - loss: 0.1728 - acc: 0.9326 - precision: 0.9427 - recall: 0.8866 - auc: 0.9866 - val_loss: 0.3393 - val_acc: 0.8182 - val_precision: 0.7500 - val_recall: 0.7500 - val_auc: 0.9643
    Epoch 16/25
    3/3 [==============================] - 12s 4s/step - loss: 0.1656 - acc: 0.9164 - precision: 0.9167 - recall: 0.8835 - auc: 0.9898 - val_loss: 0.5802 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.9286
    Epoch 17/25
    3/3 [==============================] - 12s 5s/step - loss: 0.1326 - acc: 0.9212 - precision: 0.9125 - recall: 0.8876 - auc: 0.9910 - val_loss: 0.3503 - val_acc: 0.9091 - val_precision: 1.0000 - val_recall: 0.7500 - val_auc: 0.9286
    Epoch 18/25
    3/3 [==============================] - 12s 4s/step - loss: 0.0785 - acc: 0.9764 - precision: 1.0000 - recall: 0.9454 - auc: 1.0000 - val_loss: 0.4082 - val_acc: 0.7273 - val_precision: 0.6000 - val_recall: 0.7500 - val_auc: 0.9286
    Epoch 19/25
    3/3 [==============================] - 12s 4s/step - loss: 0.0333 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - val_loss: 0.7175 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.9286
    Epoch 20/25
    3/3 [==============================] - 12s 4s/step - loss: 0.0477 - acc: 0.9814 - precision: 0.9566 - recall: 1.0000 - auc: 1.0000 - val_loss: 0.6752 - val_acc: 0.8182 - val_precision: 0.6667 - val_recall: 1.0000 - val_auc: 0.9286
    Epoch 21/25
    3/3 [==============================] - 12s 4s/step - loss: 0.0131 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - val_loss: 0.4352 - val_acc: 0.7273 - val_precision: 0.6000 - val_recall: 0.7500 - val_auc: 0.9286
    Epoch 22/25
    3/3 [==============================] - 12s 4s/step - loss: 0.0064 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - val_loss: 0.5731 - val_acc: 0.8182 - val_precision: 0.7500 - val_recall: 0.7500 - val_auc: 0.8393
    Epoch 23/25
    3/3 [==============================] - 12s 4s/step - loss: 8.2696e-04 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - val_loss: 0.8376 - val_acc: 0.8182 - val_precision: 0.7500 - val_recall: 0.7500 - val_auc: 0.8571
    Epoch 24/25
    3/3 [==============================] - 12s 4s/step - loss: 5.4741e-04 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - val_loss: 1.0772 - val_acc: 0.8182 - val_precision: 0.7500 - val_recall: 0.7500 - val_auc: 0.8393
    Epoch 25/25
    3/3 [==============================] - 12s 4s/step - loss: 9.1163e-05 - acc: 1.0000 - precision: 1.0000 - recall: 1.0000 - auc: 1.0000 - val_loss: 1.2346 - val_acc: 0.8182 - val_precision: 0.7500 - val_recall: 0.7500 - val_auc: 0.7857
    ---------------------------------------------------------
                      Classification Report
    
                  precision    recall  f1-score   support
    
               0       1.00      0.73      0.85        15
               1       0.73      1.00      0.85        11
    
        accuracy                           0.85        26
       macro avg       0.87      0.87      0.85        26
    weighted avg       0.89      0.85      0.85        26
    
    ---------------------------------------------------------
    


![png](/images/output_59_1.png)



![png](/images/output_59_2.png)



![png](/images/output_59_3.png)



![png](/images/output_59_4.png)



![png](/images/output_59_5.png)



![png](/images/output_59_6.png)


    ------------------------------------------------------------
    1/1 [==============================] - 3s 3s/step - loss: 1.2990 - acc: 0.8462 - precision: 0.7333 - recall: 1.0000 - auc: 0.9152
    loss score: 1.2989661693572998
    accuracy score: 0.8461538553237915
    precision score: 0.7333333492279053
    recall score: 1.0
    auc score: 0.9151515364646912
    
    Time to run cell: 310 seconds
    

We seem to be getting an 85% accuracy with our model and a recall for the COVID class of 100%. However, when it comes to classifying our spectrogram images with a basic model we've created, the dataset we are working with is too small - only containing around 16 different patients with 121 segmented audio samples. We can also see that our model begins to overfit after the 8th epoch. Let's get more audio data from the coughvid dataset and combine it with the virufy dataset.

## Obtaining CoughVid audio data

We understand that in order to create a model that we ca rely on to give us trustworthy results, we must include more data for our model to look at - or run the risk of having our model overfit to the data. We will begin exploring the CoughVid dataset. Again, we must unzip the saved file stored in our Google Drive in order to access the CoughVid audio data. Note that CoughVid is a crowdsourced dataset, gathered from the CoughVid app, which can be found in the link below. This dataset contains over 20,000 different types of cough audio samples.

link to CoughVid app: https://coughvid.epfl.ch/


```python
# Viewing the distribution of age in coughvid dataset
fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(data=coughvid_df, x='age', bins=10, kde=True)
plt.axvline(x=coughvid_df['age'].mean(), c = 'r', label='Mean Age')
plt.title('Distribution of Age')
plt.legend()
plt.show()
```


![png](/images/output_63_0.png)


### Column information

>- **uuid**: The address of the associated audio and json file for a patient.
- **datetime**: Timestamp of the received recording in ISO 8601
format.
- **cough_detected**: Probability that the recording contains cough sounds, according to the automatic detection algorithm
that was used by Orlandic et al.
- **latitude**: Self-reported latitude geolocation coordinate with reduced precision.
- **longitude**: Self-reported longitude geolocation coordinate with reduced precision.
- **age**: Self-reported age value.
- **gender**: Self-reported gender.
- **respiratory_condition**: If the patient has other respiratory conditions (self-reported).
- **fever_muscle_pain**: If the patient has a fever or muscle pain (self-reported).
- **status**: The patient self-reports that has been diagnosed with COVID-19 (COVID), that has symptoms but no diagnosis (symptomatic), or that is healthy (healthy).

>Within the next set of columns, it is important to know that 3 expert pulmonologists were each assigned with revising 1000 recordings to enhance the quality of the dataset with clinically validated information. They selected one of the predefined options to each of the following 10 items:



>**Categorical Columns**:
- **quality**: quality of the recorded cough sound.
  - values: {good, ok, poor, no_cough} 
- **cough_type**: Type of the cough.
  - values:  {wet, dry, unknown}
- **diagnosis**: Impression of the expert about the condition of the patient. It can be an upper or lower respiratory tract
infection, an obstructive disease (Asthma, COPD, etc), COVID-19, or a healthy cough.
  - values: {upper_infection, lower_infection, obstructive_disease, COVID-19, healthy_cough}
- **severity**: Impression of the expert about the severity of the cough. It can be a pseudocough from a healthy patient, a mild or severe cough from a sick patient, or unknown if the expert can’t tell.
  - values: {pseudocough, mild, severe, unknown}

>**Boolean Columns**:
- **dyspnea**: Presence of any audible dyspnea.
- **wheezing**: Presence of any audible wheezing.
- **stridor**: Presence of any audible stridor.
- **choking**: Presence of any audible choking.
- **congestion**: Presence of any audible nasal congestion.
- **nothing**: Nothing specific is audible.


We see that the majority of our data have missing values in the expert columns. This is expected because they each reviewed only 1000 audio files, therefore the majority of these values should be missing. Also note that about 15% of the recordings were labeled by all three reviewers, so that Orlandic et al. could assess the level of agreement among the pulmonologists.


Our Reasoning for setting the threshold of cough_detection >= 0.8:

According to Orlandic et al., "the ROC curve of the cough classifier is displayed below, which users of the COUGHVID database can consult to set a cough detection threshold that suits their specifications. As this figure shows, only 10.4% of recordings with a cough_detected value less than 0.8 actually contain cough sounds. Therefore, they should be used only for robustness assessment, and not as valid cough examples."


## Inspecting, Scrubbing, & Exploring

![auc_roc_curve_cough_detection.jpg](data:image/jpeg;base64,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)

Figure from: Lara Orlandic, Tomas Teijeiro, & David Atienza. (2021). The COUGHVID crowdsourcing dataset: A corpus for the study of large-scale cough analysis algorithms (Version 2.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4498364

This dataset required a lot of scrubbing to retrieve the data we needed to work with. Along with using the 'cough_detected' column > 0.8 as a filter for choosing the correct audio to model off of, we also had to remove rows where the labels were null or labeled as 'symptomatic'. The values within the 'status' column are self-reported by the patient. The values include: COVID, symptomatic, and healthy. In order for us to properly train a model, we cannot use the symptomatic audio files. People who report their status as symptomatic indicating that they do have symptoms similar to COVID-19 symptoms, but they have not been diagnosed as being COVID-19 positive. So while some of the patients who report as symptomatic may in fact have COVID-19, we must discard this class because it is not definitive.


### Moving all audio files into new directory

Since the unzipped public dataset not only contains our desired audio files, but also related .json files and a .csv file containing the metadata for the entire dataset, we are going to move the audio files into a new directory so we don't have to worry about any other files in the directory while working with the target audio files.

### Separating healthy and covid audio via separate directories

We filter our audio files by placing the audio file into a new folder, either 'pos' or 'neg', by targeting the uuid column and moving the audio from its address into  the proper folder.

### Finding Duration for audio files

Like we mentioned earlier, it's important to make sure the audio clips are all the same length. Below, we create a dataframe that contains the duration of each audio file in our folder that contains the audio files that match our 'uuid' column in our scrubbed dataset.


```python
sns.histplot(duration_df, x='duration')
plt.title('Distribution of Duration of Audio Files');
```


![png](/images/output_80_0.png)


Above, we see that the majority of our audio in the target coughvid data is around 10 seconds long. However, there is a lot of data that is less than 10 seconds. Our next step is to combine the virufy data with the coughvid data, and then extend the audio files that are less than 10 seconds. Below, we find that the 99th percentile is approximately 10.02 seconds. We will set 10.02 as the maximum duration for our audio files.

### Converting .webm audio to .ogg audio

Many of the files from the coughvid dataset are .webm files - which are video files. In order for librosa to be able to extend the duration of these files, we must first convert them into .ogg files so they will be compatible with the librosa library. These steps took a a considerable amount of time to run, so we also created a copy of these folders and saved them in Google Drive so we would not have to run the below cells every time we wanted to access these files.

### Adding Silent length to Target audio folders

Now that we've converted the .webm files to .ogg files, our next step is to extend the length of these files and combine extended virufy audio into our new folders. Similar to how we want to make sure CNN models are fed images of the same image shape, we want to make sure our audio files are also the same shape. Time is an axis on a spectrogram, therefore we must make sure the time in each audio file is equal.

### Extending length of virufy audio

Since we are combining the CoughVid dataset with the Virufy dataset, we must make sure tha the Virufy set is the same length in terms of time as the rest of the CoughVid dataset. We implement the librosa library and zero pad the audio files until their duration is equal to 10.02 seconds - the same as our extended CoughVid audio.

### Creating and saving mel-spectrograms

Now that we've extended all target audio files and combined the virufy data with the Coughvid data, it is time to create and save mel-spectrogram images that our model will be training on. For more information on how to create mel-spectrograms via the librosa library, visit this link: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html

## Modeling

### Preparing Image Data Generators

Now that we've gone through the preprocessing steps to make our audio the same length and created mel-spectrograms, we now must create train, test, and validation folders to split the spectrogram files according to class. Again, we use the splitfolders library to help us. We should also note that we have manually zipped the folder containing all the spectrograms before continuing. This way, we can extract the contents of the zipped folder within our Google Drive, allowing for much quicker access to the spectrogram images. 

We then create train, test, and validation folder sets, and have our newly created Image Data Generator flow the data from each of these directories into respective iterators. We implement some basic augmentation on the training set to try and protect against the heavy class imbalance. We also apply our class weights function to add to our fit_plot_report_gen() function. 

Note here that unlike our Virufy dataset, the combined dataset is heavily imbalanced, with most images belonging to the "Healthy" class. Notice that when we created the training image data generator, we added augmentation along with pixel normalization as an effort to help the model see "more images" because of the class imbalance. 


```python
# Our classes are extremely unbalanced
class_weights_dict = make_class_weights(train_spectro_gen)
```

    {0: 0.5515752032520326, 1: 5.347290640394089}
    

### Creating a model function for spectrograms


```python
spectro_fpr = fit_plot_report_gen(spectro_model1, train_spectro_gen, 
                                  test_spectro_gen, val_spectro_gen, epochs=5, 
                                  class_weights=class_weights_dict)
```

    Epoch 1/5
    136/136 [==============================] - 433s 3s/step - loss: 0.1075 - acc: 0.8786 - precision: 0.0000e+00 - recall: 0.0000e+00 - auc: 0.4752 - val_loss: 0.0674 - val_acc: 0.9079 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - val_auc: 0.5492
    Epoch 2/5
    136/136 [==============================] - 421s 3s/step - loss: 0.0741 - acc: 0.9030 - precision: 0.0000e+00 - recall: 0.0000e+00 - auc: 0.4876 - val_loss: 0.0517 - val_acc: 0.9079 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - val_auc: 0.5493
    Epoch 3/5
    136/136 [==============================] - 427s 3s/step - loss: 0.0706 - acc: 0.9110 - precision: 0.0000e+00 - recall: 0.0000e+00 - auc: 0.5100 - val_loss: 0.0636 - val_acc: 0.9079 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - val_auc: 0.5325
    Epoch 4/5
    136/136 [==============================] - 423s 3s/step - loss: 0.0725 - acc: 0.9046 - precision: 0.0000e+00 - recall: 0.0000e+00 - auc: 0.5530 - val_loss: 0.0721 - val_acc: 0.9079 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - val_auc: 0.4843
    Epoch 5/5
    136/136 [==============================] - 420s 3s/step - loss: 0.0736 - acc: 0.9020 - precision: 0.0000e+00 - recall: 0.0000e+00 - auc: 0.5067 - val_loss: 0.0634 - val_acc: 0.9079 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - val_auc: 0.5759
    ---------------------------------------------------------
                      Classification Report
    
                  precision    recall  f1-score   support
    
               0       0.91      1.00      0.95      1126
               1       0.00      0.00      0.00       118
    
        accuracy                           0.91      1244
       macro avg       0.45      0.50      0.48      1244
    weighted avg       0.82      0.91      0.86      1244
    
    ---------------------------------------------------------
    

    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


![png](/images/output_96_2.png)



![png](/images/output_96_3.png)



![png](/images/output_96_4.png)



![png](/images/output_96_5.png)



![png](/images/output_96_6.png)



![png](/images/output_96_7.png)


    ------------------------------------------------------------
    39/39 [==============================] - 78s 2s/step - loss: 0.0635 - acc: 0.9051 - precision: 0.0000e+00 - recall: 0.0000e+00 - auc: 0.5751
    loss score: 0.0634758397936821
    accuracy score: 0.9051446914672852
    precision score: 0.0
    recall score: 0.0
    auc score: 0.5750632882118225
    
    Time to run cell: 2285 seconds
    

Our model is having a very difficult time differentiating between our classes. Even though we used class weights, our model is only guessing every label as the majority label (healthy). So while our model is giving us an 91% accuracy, it is basically useless. Let's try and give our model more COVID images to look at. In our next step, we will create augmented images of the minority class from the training set and balance the training set with those augmented images.



### Oversampling with Image Augmentation Manipulation

We are going to attempt to address the class imbalance by creating augmented images of the minority class spectrograms and combine these images with our training folder to create a faux-balanced dataset. To do this, we will:
- create a copy of our training set audio folder containing both classes
- remove all majority class images
- create a new ImageDataGenerator with augmentation
- manipulate the generator by stating the batch size as the amount in our minority class (406)
- create a new folder to store iterations of the augmented data until the minority folder number of images is equal to the number of images in the majority folder 
- create another copy of our audio folder containing both classes
- add the augmented images into the minority folder of the training set
- create a new image data generator and iterators for our new data to flow through
- Fit a model to the new training set


```python
spectro_oversamp_fpr = fit_plot_report_gen(spectro_model3, train_spec_gen, test_spec_gen, val_spec_gen, 
                                           epochs=10)
```

    Epoch 1/10
    246/246 [==============================] - 335s 1s/step - loss: 0.0718 - acc: 0.5370 - precision: 0.6727 - recall: 0.1092 - auc: 0.6404 - val_loss: 0.0843 - val_acc: 0.9128 - val_precision: 0.8000 - val_recall: 0.0702 - val_auc: 0.4806
    Epoch 2/10
    246/246 [==============================] - 331s 1s/step - loss: 0.0485 - acc: 0.7717 - precision: 0.9553 - recall: 0.5679 - auc: 0.8598 - val_loss: 0.0331 - val_acc: 0.9095 - val_precision: 0.6667 - val_recall: 0.0351 - val_auc: 0.5297
    Epoch 3/10
    246/246 [==============================] - 330s 1s/step - loss: 0.0289 - acc: 0.8845 - precision: 0.9844 - recall: 0.7825 - auc: 0.9318 - val_loss: 0.0318 - val_acc: 0.9079 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - val_auc: 0.6028
    Epoch 4/10
    246/246 [==============================] - 330s 1s/step - loss: 0.0275 - acc: 0.8805 - precision: 0.9809 - recall: 0.7754 - auc: 0.9468 - val_loss: 0.0334 - val_acc: 0.9079 - val_precision: 0.5000 - val_recall: 0.0526 - val_auc: 0.5786
    Epoch 5/10
    246/246 [==============================] - 329s 1s/step - loss: 0.0214 - acc: 0.9052 - precision: 0.9859 - recall: 0.8229 - auc: 0.9693 - val_loss: 0.0406 - val_acc: 0.8853 - val_precision: 0.0625 - val_recall: 0.0175 - val_auc: 0.6374
    Epoch 6/10
    246/246 [==============================] - 327s 1s/step - loss: 0.0132 - acc: 0.9416 - precision: 0.9921 - recall: 0.8880 - auc: 0.9882 - val_loss: 0.0500 - val_acc: 0.8643 - val_precision: 0.2000 - val_recall: 0.1579 - val_auc: 0.6153
    Epoch 7/10
    246/246 [==============================] - 327s 1s/step - loss: 0.0088 - acc: 0.9671 - precision: 0.9928 - recall: 0.9408 - auc: 0.9945 - val_loss: 0.0534 - val_acc: 0.8627 - val_precision: 0.2083 - val_recall: 0.1754 - val_auc: 0.6336
    Epoch 8/10
    246/246 [==============================] - 327s 1s/step - loss: 0.0052 - acc: 0.9814 - precision: 0.9974 - recall: 0.9657 - auc: 0.9980 - val_loss: 0.0574 - val_acc: 0.8788 - val_precision: 0.0909 - val_recall: 0.0351 - val_auc: 0.6200
    Epoch 9/10
    246/246 [==============================] - 326s 1s/step - loss: 0.0024 - acc: 0.9902 - precision: 0.9983 - recall: 0.9821 - auc: 0.9997 - val_loss: 0.0931 - val_acc: 0.8772 - val_precision: 0.2564 - val_recall: 0.1754 - val_auc: 0.6208
    Epoch 10/10
    246/246 [==============================] - 324s 1s/step - loss: 0.0021 - acc: 0.9949 - precision: 0.9990 - recall: 0.9905 - auc: 0.9994 - val_loss: 0.0813 - val_acc: 0.8805 - val_precision: 0.2258 - val_recall: 0.1228 - val_auc: 0.6396
    ---------------------------------------------------------
                      Classification Report
    
                  precision    recall  f1-score   support
    
               0       0.91      0.96      0.94      1126
               1       0.27      0.14      0.18       118
    
        accuracy                           0.88      1244
       macro avg       0.59      0.55      0.56      1244
    weighted avg       0.85      0.88      0.87      1244
    
    ---------------------------------------------------------
    


![png](/images/output_100_1.png)



![png](/images/output_100_2.png)



![png](/images/output_100_3.png)



![png](/images/output_100_4.png)



![png](/images/output_100_5.png)



![png](/images/output_100_6.png)


    ------------------------------------------------------------
    39/39 [==============================] - 77s 2s/step - loss: 0.0879 - acc: 0.8834 - precision: 0.2712 - recall: 0.1356 - auc: 0.6228
    loss score: 0.08785425126552582
    accuracy score: 0.8834404945373535
    precision score: 0.2711864411830902
    recall score: 0.1355932205915451
    auc score: 0.622836172580719
    
    Time to run cell: 3448 seconds
    

## Interpretation

Our model had an accuracy of 88% with a recall of around 13.5%. This model is performing better than our first model that used the spectrograms without augmentation, in  terms of recall scoring (the first model had a recall rate of 0). This model is still having a difficult time differentiating the correct class based off of the spectrogram images alone. Our AUC improved slightly improved, however it is still too low of a score to trust this model's ability to classify COVID-19 in a patient's cough audio.

#### Possible reasons our models are struggling to differentiate between classes:

1. The silence in audio files may be introducing ambiguity into the models, which could be interfering with our model's accuracy and ability to differentiate between the classes.
2. Our model could struggling with identifying the different classes due to the heavy class imbalance.
3. The labels that were marked in the coughvid dataset 'status' column were self-reported, so there could be noise in the labels.
4. The model may not be able to find any patterns in the spectrograms we created from the audio files.
5. Our model may not be complex enough to find any patterns in our spectrograms.
6. The difference in duration between audio files may be too great; some are 2 second audio files with 8 seconds of zero padding, which could be affecting the model's ability to correctly classify.
7. More data may be required.

## Recommendations Section

Taking advantage of the splitfolders library is a great and easy way to create train, test, and validation folders for any type of classification data - as long as the classes are predefined in their own folders. 

I highly recommend using CLAHE as a preprocessing technique if you're working with images like x-rays or MRI scans. CLAHE is able to provide enough contrast to the image without overamplifying the intensity of the pixels, providing more 'depth' within each image. It is a great tool if the goal of your project involves detection and/or recognition between classes.

The librosa library is filled with tools to help assist with manipulating audio, extracting features, and creating different plots of different audio transformations. If you end up working with audio data, I recommend implementing the librosa library to help explore and create valuable features and plots from the audio data.

When working with spectrograms created from human audio, taking the log of the amplitude and converting it to decibels will give your model more to look at, and allow it to learn more from each image. Since we are working with coughing audio, converting the frequency to the mel-scale allows us to peer more into the tonal relationship of the frequencies.

While my audio model still needs more work and further exploration, I recommend health companies to invest in obtaining high quality COVID-19 positive audio data with expert-confirmed labels. Having publicly accessible high quality data would be the key to helping prevent the further spread of COVID-19, and possible future variants/strains of coronavirus. 



## Conclusion

The cough data we retrieved is currently (early 2021) sparse and hard to come by. COVID-19 audio datasets of high quality with laboratory-diagnosed labels are rare and many institutions that are working on cleaning their data for their own models have yet to make their data available to the public (such as University of Cambridge, Massachussetts Institute of Technology, NIH, etc.). There are different ways other researchers have gone about using Convolutional Neural Networks with cough audio data, such as: spectrograms, Mel-Frequency cepstral coefficients, audio feature extraction, and mel-spectrograms why trying to classify the COVID-19 cough audio. While I was unable to get any real traction when it came to classifying COVID through the use of spectrograms, I learned a lot about manipulating auditory data. Even though the models did not perform to my expectation, I'm sure the knowledge I've gained from this project could be useful when identifying other health-related events, such as detecting heart diseases based off of electrocardiographies or heartbeat audio. While I am nowhere near done with this project, I can say I've fully enjoyed the entire process.

## Future Research

While I have only been able to explore a few different methods when creating spectrograms, I have a lot of different options in front of me in terms of identifying further ways to tackle this problem. Finding other coughing audio through other sources could prove to be beneficial, especially if the audio is high quality with laboratory confirmed labels. Using different parameters when creating the spectrograms could also help the model's recognition ability. We could also try other audio imaging techniques like Mel-Frequency Cepstral Coefficients (MFCCs), or even try feature extraction to try and find key features that impact the detection efficiency when using the model to diagnose the audio.  

Another tactic that I will try will involve setting the audio files to a lower time duration before creating the spectrograms. I've realized that some of the audio (like the virufy audio) is less than 2 seconds in duration. By zero padding 8 seconds to the audio file, there may not be much information the model can really work with. So my idea is to clip the silence off the ends of the audio files, then cut each audio file in half that is greater than 6 seconds in duration. Another similar option would be to create 2 second segments for each audio file, and zero pad audio files that are less than 2 seconds in duration. This way, I'd be reducing the width of the spectrograms and giving the models more to look at. 

## References

1. M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.


2. Lara Orlandic, Tomas Teijeiro, & David Atienza. (2020). The COUGHVID crowdsourcing dataset: A corpus for the study of large-scale cough analysis algorithms (Version 1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4048312


3. Chaudhari, Gunvant, et al. "Virufy: Global Applicability of Crowdsourced and Clinical Datasets for AI Detection of COVID-19 from Cough." arXiv preprint arXiv:2011.13320 (2020).


4. Hosseiny M, Kooraki S, Gholamrezanezhad A, Reddy S, Myers L. Radiology Perspective of Coronavirus Disease 2019 (COVID-19): Lessons From Severe Acute Respiratory Syndrome and Middle East Respiratory Syndrome. AJR Am J Roentgenol2020;214:1078-82. doi:10.2214/AJR.20.22969 pmid:32108495
