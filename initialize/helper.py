import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import cv2
import random
import time

# loading functions

def decolor(input, batch_size): #used to convert multiple grayscale images
  imgs = np.zeros((batch_size, 1, 224, 224))

  for i in range(batch_size):
    x = np.dot(input[i,:,:3], [0.2989, 0.5870, 0.1140])
    imgs[i,0,:,:] = x
  return imgs

def truth(image): #used to convert to grayscale for one image
  x = np.dot(image[:,:,:], [0.2989, 0.5870, 0.1140])
  return x

def combine(input_gray, input_colour, batch_size): #combines labspace gray and colored images into a single set of images
    imgs = np.zeros((batch_size, 3, 224, 224))

    for i in range(batch_size):
      imgs[i, 0,:,:] = np.transpose(input_gray[i])
      imgs[i, 1:,:,:] = np.transpose(input_colour[i])

    return imgs

def grayify(input_gray, batch_size): #used to create input data, cant be previewed
    imgs = np.zeros((batch_size, 3, 224, 224))

    for i in range(batch_size):
      imgs[i,0,:,:] = input_gray[i].transpose()

    return imgs

def pleaseWork(images, multiple=True, randomize=True): #previews a random image from a set
  if multiple==True:
    if randomize:
      x = random.randint(0,len(images))
    else:
      print("Pick an index between", 0,"and", len(images))
      x = int(input()) 
    pastry = images[x].transpose()
    pastry = pastry.astype("uint8")
    pastry = cv2.cvtColor(pastry, cv2.COLOR_LAB2RGB)
    ruby = truth(pastry) 
    a = plt.imshow(pastry)
    plt.show()
    b = plt.imshow(ruby, cmap=plt.get_cmap('gray'), vmin=0, vmax=255) 
    plt.show()
  else:
    pastry = images.transpose()
    pastry = pastry.astype("uint8")
    pastry = cv2.cvtColor(pastry, cv2.COLOR_LAB2RGB)
    plt.imshow(pastry)  

def showTrials(outputs):
  results = []
  count_epoch = 1
  count_image = 1
  num_items = len(outputs)*len(outputs[0])

  for b in range(len(outputs[0])): #batch size
    for i in range(len(outputs)): #num of epochs
      a = outputs[i][b].detach().numpy() #tensor to array
      results.append(a) 

  for i in results:

    if count_epoch < len(outputs)-1:
      print('Epoch:',count_epoch, "|","IMAGE", count_image)
      count_epoch+=1
      c = pleaseWork(i, multiple=False)
      plt.show(c)
    elif count_epoch==len(outputs)-1:
      print("Ground Truth |", "IMAGE", count_image)
      count_epoch+=1
      gem = i
      c = pleaseWork(i, multiple=False)
      plt.show(c)
    elif count_epoch==len(outputs):
      print("Grayscale Input |", "IMAGE", count_image)
      c = gem.transpose()
      c = c.astype("uint8")
      c = cv2.cvtColor(c, cv2.COLOR_LAB2RGB)
      c = np.dot(c[:,:,:], [0.2989, 0.5870, 0.1140])
      b = plt.imshow(c, cmap=plt.get_cmap('gray'), vmin=0, vmax=255) 
      plt.show(b)
      count_epoch=1
      count_image+=1

def showAll(data):
  for i in range(500): #show all images
    a = pleaseWork(data[i], multiple=False)
    plt.show(a)

#more functions

all_batch_num = 10
iterations = int((10000/all_batch_num)/10)

def combineAll(input_gray, input_colour, interval): #combines labspace gray and colored images into a single set of images
    imgs = np.zeros((all_batch_num, 3, 224, 224))

    for i in range(all_batch_num):
      imgs[i, 0,:,:] = input_gray[i+interval].transpose()
      imgs[i, 1:,:,:] = input_colour[i+interval].transpose()

    return imgs

def grayifyAll(input_gray, interval): #used to create input data, cant be previewed
    imgs = np.zeros((all_batch_num, 3, 224, 224))

    for i in range(all_batch_num):
      imgs[i,0,:,:] = input_gray[interval+i].transpose()

    return imgs

def showAllInput(data):
  for i in data: #show all images
    ruby = i.astype("uint8").transpose()
    ruby = cv2.cvtColor(ruby, cv2.COLOR_LAB2RGB)
    ruby = truth(ruby)
    b = plt.imshow(ruby, cmap=plt.get_cmap('gray'), vmin=0, vmax=255) 
    plt.show(b)    

def showGoodTrials(results):
  n = int(len(results)/3)
  for i in range(n):
    a = i*3
    sliced = results[a:a+3]
    showTrials(sliced)

