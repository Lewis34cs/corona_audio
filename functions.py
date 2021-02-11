import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys, glob, shutil
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.utils import class_weight
from skimage.segmentation import mark_boundaries
import time
from PIL import Image as Im
import cv2
import shutil
from google.colab import files
import splitfolders
from tqdm import tqdm
from pydub import AudioSegment
import ffmpeg
import soundfile as sf
import lime
from lime import lime_image
import warnings
from zipfile import ZipFile
import librosa

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.applications import VGG19, InceptionV3, DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow_addons as tfa
import librosa.display as ldp
import IPython.display as ipd

def folder_contents(source_folder=os.path.abspath(os.curdir)):
    """
    Definition:
    Displays the contents within a source folder.

    Parameter:
    source_folder: default = os.path.abspath(os.curdir). Displays contents within
    the source folder.

    Returns:
    Prints the address of the source_folder and the contents within.
    """

    print(f"Contents of {source_folder}:\n")
    display(sorted(os.listdir(source_folder)))

def time_count(start, end):
  """
  Definition:
  Simple function that prints how many seconds it took to run from the 'start'
  variable to the 'end' variable.

  Args:
  start: Required. Usually a time.time() variable at the beginning of a cell.
  end: Required. Usually a time.time() variable at the end of a cell.

  Returns:
  Prints the difference in time between the 'start' and 'end' variables.
  """
  
  print(f"Time to run cell: {int(end-start)} seconds")

def ensure_filepath(f_path):
  """
  Definition:
  Ensures the filepath has a slash at the end of the string. 

  Args:
  f_path: Required. The string filepath.

  Returns:
  The filepath with a slash at the end.
  """
  
  if f_path[-1] != '/':
    f_path = f_path + '/'
  return f_path

def num_items(folder, targets=['COVID', 'NORMAL', 'Viral Pneumonia'], verbose=True):
  """
  Definition:
  Adds up the number of files within each target folder and prints out the target
  folder name along with the total number of items in the folder. Also prints 
  the folder's directory path.

  Args:
  folder: Required. A filepath containing the data you wish to count.
  targets: default = ['/COVID', '/NORMAL', '/Viral Pneumonia']. Specify the target
  folders you wish to see how many items are contained within each.

  Returns:
  Prints out directory path, target folder names, and total number of items
  within each folder. Returns the 'num_items' variable.
  """
  if verbose:

    num_items = 0
    # Printing the folder address we are targeting
    print(folder)
    # the target is each subfolder within our folder 
    # (the default is NORMAL and PNEUMONIA)
    for target in targets:
        # counts the number of images within each target
        num_in_target = len(os.listdir(folder+target))
        # add the total number of images for each folder
        num_items += num_in_target
        print(f"Number of items in {target}: {num_in_target}")
    print(f"Total number of items: {num_items}")
    return num_items
  
  else:
    num_items = 0
    for target in targets:
        # counts the number of images within each target
        num_in_target = len(os.listdir(folder+target))
        # add the total number of images for each folder
        num_items += num_in_target
    return num_items

def folder_check(new_dir, orig_dir=None, subfolders=['train', 'test', 'val'], 
                 check=0):
  """
  Definition:
  This funcion works best if splitfolders library was used to split data into 
  ratioed folders. The 'root_dir' must contain the folders defined in the 
  'subfolders' list. Counts the number of items for each class within 
  each subfolder. Checks the number of images to the original folder (optional).

  Args:
  root_dir: Required. Address of newly created folder.
  orig_dir: default = None. Address of original folder where images originated from.
  subfolders: default = ['/train', '/test', '/val']. Subfolders within the root_dir
  that contain the images.
  check: default = 1.

  Returns:
  Prints the total image count for all subfolders. Optionally prints boolean value
  if the total image count is equal to the total count of the orig_dir.
  """
  count_images = 0
  
  for folder in subfolders:
    targets = os.listdir(new_dir + folder)
    count_images += num_items(folder=new_dir + folder + '/', targets=targets)
    print()
  if check:
    total_image_number = num_items(orig_dir)
    print(f"\nAre the image numbers equal?\n{total_image_number == count_images}")

  print(f"\nTotal image count: {count_images}")

def create_ttv(root_dir):
  """
  Definition:
  Appends the string of each address within the 'root_dir' variable to the list 
  'ttv_list', sorts them by ABC order, and returns the list.

  Args:
  root_dir: Required. Specify a directory that contains train, test, and validation
  folders.

  Returns:
  The address of each train, test, and validation folder within the specified
  directory.
  """
  
  # creating an empty list to hold addresses
  ttv_list = []
  for folder in os.listdir(root_dir):
    # creating and adding the string of each address to 'ttv_list'
    ttv_list.append(root_dir + folder + '/')
  # Sorting the list alphabetically to capture the correct addresses for each variable
  ttv_list.sort()
  test_folder = ttv_list[0]
  train_folder = ttv_list[1]
  val_folder = ttv_list[2]
  # returning the variables that each contain an address 
  return train_folder, test_folder, val_folder

def batch_sizes(folder_list=[]):
  """
  Definition:
  Creates batch sizes for each folder by counting the total number of items in 
  each folder. Appends these numbers to a list "batch" and returns the list. 
  Works best with the create_ttv() function.

  Args:
  folder_list: default = []. Pass in a list of folder directories that contain
  items you wish to be counted for batch sizes. 

  Returns:
  A list containing the batch sizes for each folder that was passed in.

  """
  batch = []
  for folder in folder_list:
    targets = os.listdir(folder)
    batch.append(num_items(folder, verbose=0, targets=targets))
  return batch

def random_image(X, y, verbose=0):
  """
  Definition:
  Selects a random number (i) within the range of the length of X. Then prints 
  the class label associated with that random number (i) along with the image
  associated with the specific X[i] array.

  Args:
  X: an np.array
  y: labels associated with the np.array

  Returns:
  Prints the class along with the y label, and displays the randomly selected image.
  """

  # Getting a random number within the range of our X variable
  i = np.random.choice(range(len(X)))
  # Determining what the label is and printing appropriate class
  if verbose:

    if y[i] == 0:
      print(f"COVID-19 : Class {y[i]}")
    elif y[i] == 1:
      print(f"NORMAL : Class {y[i]}")
    else:
      print(f"Viral Pneumonia : Class {y[i]}")
  else:
    if y[i] == 0:
      print(f"Class {y[i]}")
    elif y[i] == 1:
      print(f"Class {y[i]}")
    else:
      print(f"Class {y[i]}")
  # Displaying the image
  display(array_to_img(X[i]))

def make_class_weights(train_gen, cls_weight='balanced', verbose=1):
  # FIX THIS AREA
  """
  Definition:
  Generates the class weights of a generator and returns a dictionary containing
  the class weights for each class present in the generator.

  Args:
  train_gen: Required. The training generator.
  cls_weight: default = 'balanced'. If ‘balanced’, class weights will be given 
  by n_samples / (n_classes * np.bincount(y)). If a dictionary is given, keys 
  are classes and values are corresponding class weights. If None is given, the 
  class weights will be uniform.
  verbose: default = 1. If verbose is true, the dictionary will also be printed.
  
  Returns:
  A dictionary containing the class weights
  """
  cwd = {}
  class_weights_list = class_weight.compute_class_weight(cls_weight,
                                            np.unique(train_gen.classes), 
                                            train_gen.classes)
  for cls in np.unique(train_gen.classes):
    cwd[cls] = class_weights_list[cls]

  if verbose:
    print(cwd)
  return cwd

def plot_history(history, metric_list=['acc', 'loss', 'precision', 'recall', 'auc']):
  """
  Definition:
  Creates a dataframe with a model.history variable. Then plots columns within the 
  dataframe if the column contains a metric within the metric list.

  Args:
  history: requires a model.history variable.
  metric_list: default = ['loss', 'acc']. Based on the metric's used in the model's 
  compiled metrics. 

  Returns:
  plots the columns that contain the metric within the metric list
  """
  # creating a dataframe of the model.history variable
  history_df = pd.DataFrame(history.history)

  with plt.style.context('seaborn'):    
    for metric in metric_list:
      history_df[[col for col in history_df.columns if metric in col]].plot(figsize=(8, 4), 
                                                                            marker='o')

      # Setting the title for each plot to be be the focused metric
      plt.title(metric.title())
      plt.grid(True)
      #sets the vertical range to [0-1]
      plt.gca().set_ylim(0, 1)
    plt.show()

def class_report_gen(model,test_gen, class_indices, cmap='Reds'):
  """
  Definition:
  Prints out a classification report by predicting y_pred using model.predict() 
  and plots a heatmap of a confusion matrix using seaborn's heatmap() function. 

  Args:
  model: Requires a model.
  X_test: Requires a test set of features.
  y_test: Requires a test set of labels.
  class_indices: default = train_set_full.class_indices. Pass through a dictionary
  that defines the classes. Must match up with the y_test labels.

  Returns:
  Prints out a classification report and a confusion matrix.  
  """
    
  # creating a title using the class_indices.items()
  title = ''
  for key, val in class_indices.items():
        title += key + ' = ' + str(val) + '    '
    
  
  # Determining the number of classes
  limit = len(np.unique(test_gen.classes))
  if limit > 2:   
    # defining our prediction for multiple classification
    y_pred = model.predict(test_gen)
    y_pred = np.argmax(y_pred, axis=1)
  else:
    # defining our predicition for binary classification
    y_pred = model.predict(test_gen)
    y_pred = np.round(y_pred)

  # Printing a classification report to see accuracy, recall, precision, f1-score
  dashes = '---'*19
  print(dashes)  
  print('                  Classification Report\n')
  print(metrics.classification_report(test_gen.classes, y_pred))
  print(dashes)
    

  # plots a normalized confusion matrix
  plt.figure(figsize=(7,6))
  conf_mat = metrics.confusion_matrix(test_gen.classes, y_pred, normalize='true')
  ax = sns.heatmap(conf_mat, cmap=cmap, annot=True, square=True)
  ax.set(xlabel='Predicted Class', ylabel='True Class')
  ax.set_ylim(limit, 0)
  ax.set(title=title)
  plt.show()

def three_callbacks(monitor='val_loss', min_delta=0, patience=0, mode='auto', 
                   f_path=None, restore_best_weights=False, save_best_only=True, 
                   save_freq='epoch', lr_patience=5, factor=0.1, cd=0):
  """
  Definition:
  Creates three variables, and then returns these variables in a list.

  1. variable 'earlystop' stops model training when the 'monitor' metric has not 
  improved past the 'min_delta' threshold after a certain amount of epochs, 
  defined by 'patience'. 

  2. variable 'checkpoint' saves model at some interval, so the model can be 
  loaded later on to continue the training from the saved state.
	
	3. variable 'reducelr' reduces learning rate after a certain number of intervals.

  Args:
  monitor: default = 'val_loss'. Quantity to be monitored during training.
  min_delta: default = 0. minimum change in the monitored quantity to be
  considered an improvement.
  patience: default = 0. The number of epochs with no improvement after which
  training will be stopped.
  mode: default = 'auto'. Defines the direction of monitor quantity. ex. If 'min', 
  training will stop after monitored quantity is no longer decreasing past the 
  defined min_delta. 
  f_path: default = ''. If left default, function will output a string indicating
  a path needs to be defined in order for a model to be saved.
  The filepath that will be created / set as the destination for saved models 
  from 'checkpoint' variable.
  restore_best_weights: default = False. Defines whether or not to restore the 
  best model weights from the epoch with the best value of the monitored quantity.
  save_best_only: default = False. If save_best_only=True, the latest best model 
  according to the quantity monitored will not be overwritten.
  save_freq: default = 'epoch'. The defined interval at which the model is saved.
  lr_patience: default = 5. Dictates how long the ReduceLROnPlateau callback must 
	wait until it can initiate learning rate decay if there is no improvement in 
  the monitored metric.
	factor: default = 0.1. Float that is multipled by the current learning rate. This action is 
	learning rate decay, and the product is the new learning rate until training stops.
	cd: default = 0. Defines how long the cooldown for ReduceLROnPlateau must wait until it
	may begin lr_patience after initiating learning rate decay. 

  Returns:
  A list named 'callbacks' containing the 'earlystop', 'checkpoint', and
  'reducelr' variable.
  """

  # Defining our early stopping func
  earlystop = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, 
                            mode=mode, restore_best_weights=restore_best_weights)
    
  #checking to make sure our path is set up. If not, the line below creates it.
  os.makedirs(f_path, exist_ok=True)
    
  # saves the current model at the specified checkpoint
  checkpoint_f_path = f_path+"model-{epoch:02d}-{"+monitor+":.3f}.hdf5"
  checkpoint = ModelCheckpoint(filepath=checkpoint_f_path, monitor=monitor, 
                              save_best_only=save_best_only, mode=mode, 
                              save_freq=save_freq)
  # reduces learning rate after a certain number of epochs if the monitored
  # parameter has not decreased more than min_delta
  reducelr = ReduceLROnPlateau(monitor=monitor, mode=mode, min_delta=min_delta, 
                              patience=lr_patience, min_lr=1e-5,factor=factor, 
                              cooldown=cd, verbose=1)
  # Store our callbacks into a list called 'callbacks'
  callbacks = [earlystop, checkpoint, reducelr]
  return callbacks

def fit_plot_report_gen(model, train_gen, test_gen, val_gen,
                    epochs=10, batch_size=32, plot_hist=1, class_weights=None,
                    callbacks=''):
  """
  Definition:
  Fits a passed in model saved in a variable 'history'. Then activates the 
  class_report function which returns a classification report and a confusion
  matrix. Finally, plots the history using the plot_history() function.

  Args:
  model: Requires a model.
  train_gen: Required. Iterator that pulls batches from the training data directory. 
  test_gen: Required. Iterator that pulls batches from the testing data directory.
  val_gen: Required. Iterator that pulls batches from the validation data directory.
  epochs: default = 10. Defines the number of passes the ML algorithm will complete.
  batch_size: default = 32. Defines the number of training examples utilized in
  one iteration before updating internal parameters.
  plot_hist: default = 1. Defines whether or not the plot_history() function will
  be executed. 
  class_weights: default = None. Pass in a dictionary containing the class weights, 
  where the keys are the classes and the values are the weights.
  callbacks: default = ''. If default, the model will fit without implementing 
  any callbacks. 

  Returns:
  history, prints classification report, confusion matrix, and plots history metrics.
  """

  start = time.time()
  # determining if model will include callbacks
  if len(callbacks) > 0:
    history = model.fit(train_gen, batch_size=batch_size,  
                      validation_data=val_gen, epochs=epochs, 
                      class_weight=class_weights, callbacks=callbacks)
  
  else:
    history = model.fit(train_gen, batch_size=batch_size,  
                      validation_data=val_gen, epochs=epochs, 
                      class_weight=class_weights)
        
  # Identifying the number of classes
  class_indices = train_gen.class_indices
  class_report_gen(model, test_gen, class_indices=class_indices)
    
  if plot_hist:
      plot_history(history) 

  dashes = '---'*20
  print(dashes)
  eval_scores = model.evaluate(test_gen)
  metric_list=['loss', 'accuracy', 'precision', 'recall', 'auc']
  for i, score in enumerate(eval_scores):
    print(f"{metric_list[i]} score: {score}")
  print()
  end = time.time()
  time_count(start, end)
  return history

def clahe_preprocessing(root_dir, new_dir):
  """
  Definition:
  from a specified directory 'root_dir', the function will loop through each
  subfolder and apply a clahe mask to each image and save that modified image in
  the 'new_dir' address.

  Args:
  root_dir: Required. The address of the directory that contains the subfolders
  holding the images.
  new_dir: Required. The directory where the modified images will be saved. This
  can be a directory that does not exist yet.

  Returns:
  Prints the time it took to run the function.
  """
  start = time.time()
  # creating our clahe mask to aply to images
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

  # Looping through the subfolders in the root_dir
  for folder in os.listdir(root_dir):
    # If the end portion of an item within the folder contains something 
    # (eg: .png, .xlsx, .txt, etc.) the function will ignore it and continue;
    # it is only looking for subfolders to loop through.
    if os.path.splitext(folder)[-1].lower():
      continue

    # applying the clahe mask and saving the updated image to the new folder
    for image in os.listdir(root_dir + folder):
      img = cv2.imread(root_dir + folder + '/' + image, 0)
      cl_img = clahe.apply(img)
      os.makedirs(new_dir + folder, exist_ok=True)
      cv2.imwrite(new_dir + folder + '/' + image, cl_img)

  end = time.time()
  time_count(start, end)

def explain_image(model, generator, top_labels=5, num_samples=1000, num_feats=3, 
                  figsize=(6,6), class_label=0):
  """
  Definition:
  Using the lime library, plots an image of a class that explains a model's 
  prediction for that class. 

  Args:
  model: Required. A CNN classifier that will make predictions on the test generator
  generator: Required. A test generator that contains labels and data of images.
  top_labels: default = 5. If an integer, ignore labels and produce explanations 
  for the X amount of labels with highest prediction probabilities, where X is
  this parameter.
  num_samples: default = 1000. The size of the neighborhood to learn the linear 
  model.
  num_feats: default = 3. The maximum number of features present in explanation.
  figsize: default = (6, 6). The size of the plotted image. Must be a tuple.
  class_label: default = 0. The class the model will be predicting. The class must
  exist, otherwise the function will throw an error.

  Returns:
  Plots an image within the specified class, showing a certain amount of the 
   model's interpreted important features that have the highest correlations 
  for that class.

  Info and some code retrieved from: 
  https://lime-ml.readthedocs.io/en/latest/lime.html
  """
  # Creating batches of images and labels from our generator
  data, label = next(generator)

  # Determining if multi or binary classification
  if len(generator.class_indices) > 2:
    # If multi
    label_class = np.argmax(label, axis=1)
  else:
    # If binary
    label_class = label
  
  # targeting a certain class for explainer
  target_class = data[label_class==class_label]

  # Randomizing the image selection from target class
  i = np.random.choice(range(len(target_class)))
  data = target_class[i]
  pred = model.predict(np.array([data]))

  # Multi or binary prediction
  if len(generator.class_indices) > 2:
    pred_class = np.argmax(pred, axis=1)
  else:
    pred_class = int(np.round(pred))
  
  # Creating lime image explainer
  lime_ie = lime_image.LimeImageExplainer()
  lime_explain = lime_ie.explain_instance(data, model.predict, 
                                          top_labels=top_labels, 
                                          num_samples=num_samples)
  
  # creates an image and a mask where the image is a 3d numpy array and the 
  # mask is a 2d numpy array that is compatible with 
  # skimage.segmentation.mark_boundaries
  temp, mask = lime_explain.get_image_and_mask(lime_explain.top_labels[0], 
                                            positive_only=False,
                                            num_features=num_feats)

  # Creating label titles
  for key, val in generator.class_indices.items():
    if pred_class == val:
      p_title = key
    if class_label == val:
      a_title = key
  

  y_ax = np.linspace(0, 8000, num=data.shape[0])

  plt.figure(figsize=figsize)
  plt.yticks(y_ax)
  plt.title(f"Predicted Class: {p_title.upper()} / Actual Class: {a_title.upper()}")
  plt.imshow(mark_boundaries(image=temp / 2 + 0.5, label_img=mask))

def move_best_models(source, new_dir, test_gen, threshold=0.9):
  """
  Definition: Moves all models within the source file if the model's accuracy
  is greater than the set threshold based off of the model.evaluate() method.
  If the model's accuracy is not higher than the threshold, the model will be 
  erased from the source.

  Args:
  source: Provide the file source to filter through models.
  new_dir: Provide new folder for the filtered models to move to. Can be a 
  filepath that does not exist.
  test_gen: Required. The generator the model will be evaluated against.
  threshold: default = 0.9. The model's accuracy must be higher than this float
  to be able to move into the 'new_dir'.

  Returns:
  Prints the evaluation metrics for each model, along with the model name if the 
  model accuracy is higher than the designated threshold.
  """

  model_list = []
  os.makedirs(new_dir, exist_ok=True)
  for f in os.listdir(source):
    try:
      model = load_model(source + f)
      eval = model.evaluate(test_gen)
      if eval[1] > threshold:
        print(f)
        model_list.append(f)
      else:
        os.remove(source + f)
    except:
      continue
  for f in model_list:
          shutil.move(source+f, new_dir)

def to_spectrogram(signal, sr, hop_length, n_fft=2048, ref=1, cmap='magma', 
                   fmax=9000, vmin=None, vmax=None, figsize=(10, 8)):
  """
  Definition:
  Plots a spectrogram of a signal.

  Args:
  signal: Required. An audio time series, usually an numpy 1 dimensional array.
  sr: Required. The sample rate of the signal.
  hop_length: Required. The amount to shift each fast fourier transform.
  n_fft: Required. The number of samples in a window per fast fourier transform.
  cmap: default = 'magma'. The color palette of the spectrogram.

  Returns:
  Plots a spectrogram via librosa.display.specshow()
  """
  stft_signal = librosa.core.stft(y=signal, hop_length=hop_length, n_fft=n_fft)
  spectrogram = np.abs(stft_signal)
  amp_to_db = librosa.amplitude_to_db(spectrogram, ref=ref)
  fig = plt.figure(figsize=figsize)
  if (vmin != None) and (vmax != None):
    ldp.specshow(amp_to_db, sr=sr, x_axis='time', y_axis='hz', cmap=cmap, 
                hop_length=hop_length, fmax=fmax, vmin=vmin, vmax=vmax)
  else:
    ldp.specshow(amp_to_db, sr=sr, x_axis='time', y_axis='hz', cmap=cmap, 
                hop_length=hop_length, fmax=fmax)
  plt.title("Spectrogram (Decibels)")
  plt.ylim(0, fmax)
  plt.colorbar(label='dB')
  # axes = fig.get_axes()
  # return fig, axes

def to_mel_spectro(signal, sr, hop_length, n_fft, cmap='magma', 
                    ref=np.max, figsize=(5, 4), vmin=None, vmax=None, n_mels=128):
  """
  Definition:
  Plots a mel-spectrogram of a signal.

  Args:
  signal: Required. An audio time series, usually an numpy 1 dimensional array.
  sr: Required. The sample rate of the signal.
  hop_length: Required. The amount to shift each fast fourier transform.
  n_fft: Required. The number of samples in a window per fast fourier transform.
  cmap: default = 'magma'. The color palette of the spectrogram.
  ref: default = np.max. Tells the librosa.power_to_db() how to scale the values.
  When ref=np.max, it will make the highest value 0, and everything else will be 
  lower than it, respectively so. Does not change the color distribution of the
  spectrogram.
  figsize: default = (10, 8). A tuple representing the desired figure size for 
  the plot.

  Returns:
  Plots a spectrogram via librosa.display.specshow()
  """

  mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, 
                                              n_fft=n_fft, n_mels=n_mels)
  spectrogram = np.abs(mel_signal)
  power_to_db = librosa.power_to_db(spectrogram, ref=ref)
  plt.figure(figsize=figsize)
  if (vmin != None) and (vmax != None):
    ldp.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap=cmap, 
                hop_length=hop_length, vmin=vmin, vmax=vmax)
  else:
    ldp.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap=cmap, 
                hop_length=hop_length)
  plt.title("Mel-Spectrogram")
  plt.colorbar(label='dB')
  plt.show()

def quality_scrub(df, target_cols = ['quality_1', 'quality_2', 'quality_3']):
  """
  Definition:
  Filters a dataframe where each target_col does not contain 'no_cough'

  Args:
  df: Required. A dataframe containing the target columns
  target_cols: default = ['quality_1', 'quality_2', 'quality_3']. 

  Returns:
  Returns a filtered dataframe where each target_col does not contain 'no_cough'
  """
  for col in target_cols:
    df = df[df[col] != 'no_cough']
  return df

def move_audio_files(df, target_col, source, dest, filetypes=['.webm', '.ogg']):
  """
  Definition:
  Provide a dataframe and target column as a unique identifier for files stored
  in the source folder. The function will move all files that match the variables
  within the target column and move them to a new destination folder.

  Args:
  df: Required. A dataframe containing a unique column identifier for files
  target_col: Required. The unique identifier used to select certain files from
  the source folder.
  source: Required. The folder containing the files we want to filter based on the
  target_col
  dest: Required. The file destination we want to copy the filtered files to.
  filetypes: default = ['.webm', '.ogg']. The types of files within the source 
  folder we are targeting.

  Returns:
  Makes a copy of the filtered files in the new destination folder.
  """
  
  start = time.time()
  # Creating new destination
  os.makedirs(dest, exist_ok=True) 
  # For each item in our target column: 
  for uuid in df[target_col]:
    # the source id equals the source variable + the item
    src = source + uuid
    try:
      # try to copy the file to the destination if it is a .webm file
      shutil.copy(src=src + filetypes[0], dst=dest)
    except:
      # If that doesn't work
      try:
        # Try to copy the file to the destination if it is an .ogg file
        shutil.copy(src=src + filetypes[1], dst=dest)
      except:
        # If that doesn't work, skip it and continue
        continue
  end = time.time()
  time_count(start, end)

def convert_audio(root_dir, new_dir):
  """
  Definition:
  For every item in the root directory, this func identifies the type of audio 
  file. If the audio file is a .webm file, the function uses the library ffmpeg
  to change the file type into an .ogg file and transfers that new file into 
  the new specified directory.

  Args:
  root_dir: Required. Declare the directory that holds the targeted files you wish
  to convert.
  new_dir: Required. Declare the new directory you wish to store the converted
  files. Does not have to exist beforehand.

  Returns:
  Creates the new directory, stores the converted files, and prints the amount 
  of time it took to run the function.
  """
  start = time.time()

  for status in os.listdir(root_dir):
    root_address = root_dir + status + '/'
    new_address = new_dir + status + '/'
    os.makedirs(new_address, exist_ok=True)

    for audio in tqdm(os.listdir(root_address), desc=status):
      extension = os.path.splitext(audio)[-1].lower()

      if extension == '.webm':
        audio_name = os.path.splitext(audio)[0]
        input = f"ffmpeg -i {root_address + audio_name}.webm {new_address + audio_name}.ogg"
        os.system(input)

      else:
        shutil.copy(src=root_address+audio, dst=new_address)

  end = time.time()
  time_count(start, end)

def get_audio_duration(root_dir):
  """
  Definition:
  Finds the duration of each audio file in a folder, uses the librosa library to
  get the time duration for each file, and appends each duration to a list called 
  duration_list. This list is then transformed into a dataframe, and then returned
  and the function output.

  Args:
  root_dir: Required. Specify the folder that holds the audio files.

  Returns:
  A dataframe that holds the duration for each audio file and prints the amount
  of time it took to run the function.
  """
  
  duration_list = []
  start = time.time()

  for status in os.listdir(root_dir):
    address = root_dir + status + '/'
    for audiofile in tqdm(os.listdir(address), desc=status):
      try:
        # setting the variable audio_time and giving it the duration of the file
        audio_time = librosa.get_duration(filename=address+audiofile)
        # appending the audio_time variable to our list
        duration_list.append(audio_time)
      except:
        # Adding 0 if we can't extract the duration. We should not get any zeros
        duration_list.append(0)

  df = pd.Series(duration_list, name='duration').to_frame()

  end = time.time()
  time_count(start, end)
  return df

def create_silence(root_dir, new_dir, max_dur=10.02):
  """
  Definition:
  Adds silence to each audio file in the specified root directory if the audio 
  file is shorter in length (in seconds) than the 'max_dur' variable. The extended
  audio files are then saved in the 'new_dir' address.

  Args:
  root_dir: Required. The directory that holds the audio files.
  new_dir: Required. The location the extended audio files will be saved to.
  max_dur: default = 10.02. The maximum duration for the audio files to be. If an
  audio file is longer than the maximum duration, the audio file will be skipped
  and not saved into the new directory.

  Returns:
  Saves the extended audio files to the new directory. Prints the time it took 
  to run the function.
  """
  
  start = time.time()

  for status in os.listdir(root_dir):
    root_address = root_dir + status + '/'
    new_address = new_dir + status + '/'
    os.makedirs(new_address, exist_ok=True)

    for audio in tqdm(os.listdir(root_address), desc=status):
      address = root_address + audio
      sig, sr = librosa.load(address)

      dur = librosa.get_duration(sig, sr=sr)
      if dur <= max_dur: 
        # Defining the amount of silence for each audio file
        silent_length = (max_dur*1000) - (dur*1000)
        silent_int = int(silent_length)

        # creating 8.4 secs of audio silence -- duration in milliseconds
        silent_segment = AudioSegment.silent(duration=silent_int)

        # Finding the extension for each file
        extension = os.path.splitext(audio)[-1].lower()

        if extension == '.mp3':
          cough_audio = AudioSegment.from_mp3(address)

        elif extension == '.ogg':
          cough_audio = AudioSegment.from_ogg(address)

        # Combining our silent segment with the audio example
        extended_cough = cough_audio + silent_segment

        # making a folder to hold our extended audio example
        output = f"{new_address + audio}" 
        #print(f"{output}\n")
        extended_cough.export(out_f=output)
      else: 
        continue

  end = time.time()
  time_count(start, end)

def create_save_spectros(root_dir, new_dir, hop_length, n_fft, sr=22050,
                         fmax=9000, ref=1, figsize=(10, 4), cmap='magma', 
                         dpi=400, vmin=None, vmax=None):
  """
  Definition:
  Takes the address defined by the variable 'root_dir' and goes through each class
  folder within that directory. For each audio file in a class directory, the 
  librosa library loads the audio file into signal and sample rate variables, then 
  plots the mel-spectrogram. We then save the spectrogram into the address of the 
  'new_dir' folder using the same end name of the audio's original address, and 
  shuts down the figure.

  Args:
  root_dir: Required. The address containing subfolders of classes and audio 
  files within those folders.
  new_dir: Required. The address you wish to save the mel-spectrograms.
  hop_length: Required. The number of samples between successive frames - 
  (e.g. the columns of a spectrogram)
  n_fft: Required. The (positive integer) number of samples in an analysis 
  window (or frame). Should be larger than hop_length.
  sr: default = 22050. The (positive integer) number of samples per second of a 
  time series.
  fmax: default = 9000. Highest frequency (in Hz).
  ref: default = 1. If scalar, the amplitude abs(S) is scaled relative to 'ref'.
  figsize: default = (10, 4). The size of the plotted spectrogram.
  cmap: default = 'magma'. The color palette of the spectrogram.
  dpi: default = 400. The resolution in dots per inch - higher values result in
  higher quality images.

  Returns:
  The time it took to run the function in seconds.
  """
  start = time.time()
  #making sure we add a forward slash to our root_dir and new_dir variables

  # For each subfolder in our root directory:
  for status in os.listdir(root_dir):

    # Create a new variable called subfolder
    subfolder = root_dir + status

    # For each audio file in our subfolder, load into signal and sample rate vars
    for audiofile in tqdm(os.listdir(subfolder), desc=status):
      signal, sr = librosa.load(subfolder + '/' + audiofile, sr=sr)

      plt.interactive(False)

      # Creating a melspectrogram
      if (vmin != None) and (vmax != None):
        mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, 
                                                    hop_length=hop_length, 
                                                    n_fft=n_fft, fmax=fmax, 
                                                    vmin=vmin, vmax=vmax)
      

      else:
        mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, 
                                                    hop_length=hop_length, 
                                                    n_fft=n_fft, fmax=fmax)
        
      # Converting the power in the ndarray of our mel_signal into decibels
      power_to_db = librosa.power_to_db(mel_signal, ref=ref)
      # Creating figure
      fig = plt.figure(figsize=figsize)
      ax = fig.add_subplot(111)
      # Hiding axes and image frame
      ax.axes.get_xaxis().set_visible(False)
      ax.axes.get_yaxis().set_visible(False)
      ax.set_frame_on(False)

      # Displaying our spectrograms
      ldp.specshow(power_to_db, sr=sr, cmap=cmap, hop_length=hop_length)
      
      # making folder to store images
      new_subfolder = new_dir + status + '/'
      os.makedirs(new_subfolder, exist_ok=True)

      #print(new_subfolder)

      # Saving each spectrogram into its respective folder
      # subfile[:-4] is a string of the subfile without the ending extension '.mp3'
      # We add the '.png' extension to the end of our new spectrogram images instead
      plt.savefig(fname=new_subfolder + audiofile[:-4] + '.png', dpi=dpi, 
                  bbox_inches='tight',pad_inches=0)
      
      # We then manually close pyplot, clear the figure, close the fig variable, 
      # and then close the figure window
      plt.close()    
      fig.clf()
      plt.close(fig)
      plt.close('all')

  # Display the time it took to run the cell in seconds
  end = time.time()
  time_count(start, end)

def reduce_audio_length(root_dir, new_dir):
  """
  Definition:
  Removes any silence at the start and end of each audio file in the specified 
  root directory if the audio. The reduced audio files are then saved in the 
  'new_dir' address.

  Args:
  root_dir: Required. The directory that holds the audio files.
  new_dir: Required. The location the extended audio files will be saved to.

  Returns:
  Saves the reduced audio files to the new directory. Prints the time it took 
  to run the function.
  """
  start = time.time()

  for status in os.listdir(root_dir):
    # making the file address
    address = root_dir + status + '/'
    os.makedirs(new_dir + status, exist_ok=True)
    for audio_file in os.listdir(address):
      # creating a tag to fit into the write_wav() function
      tag = audio_file.split('/')[-1][:-4]
      
      #loading the signal and sr for the audio file
      signal, sr = librosa.load(address + audio_file, sr=22050)

      # trimming the signal and storing it in a new variable, 'trimmed_sig'
      trimmed_sig, index = librosa.effects.trim(signal)

      # Saving our trimmed signal to the new directory
      librosa.output.write_wav(path=new_dir + status + '/' + tag + '.wav', 
                               y=trimmed_sig, sr=sr)
      
    print(f'Finished with {status}')
  end = time.time()
  time_count(start, end)

def display_images(source, amnt_to_display):
  """
  Definition:
  Define the 'source' variable by giving it a filepath containing images along with
  setting the number you wish to view through the variable 'amnt_to_display'. The
  function will plot the selected number of images within the file and display them.

  Args:
  source: Required. A filepath containing images.
  amnt_to_display: Required. The number of images you wish to display.

  Returns:
  Plots a certain amount of images from the selected filepath.
  """
  
  plt.figure(figsize=(20,10))
  cols = amnt_to_display//2
  images = os.listdir(source)[:amnt_to_display]
  for i, img in enumerate(images):
      #Opening each image from its respective filepath
      x_image = Im.open(source+img)
      # defining the position for each subplot 
      # subplot(nrows, ncols, index)
      plt.subplot(len(images) / cols + 1, cols, i + 1)
      #plotting each image in a new subplot
      plt.imshow(x_image)
      # Hiding the x and y axis tick marks
      plt.xticks([])
      plt.yticks([])
      # fitting the images closer together
      plt.tight_layout()