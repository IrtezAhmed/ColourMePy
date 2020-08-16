#code written in Google Colab - loading data


project_path = '/content/drive/My Drive/APS360_Project/'

#loading the datasets

gray = np.load('/content/drive/My Drive/APS360_Project/Dataset/grayscale/gray.npy') #all 25k grayscale images
colour1 = np.load('/content/drive/My Drive/APS360_Project/Dataset/coloured/ab1.npy') #first 10k coloured images
colour2 = np.load('/content/drive/My Drive/APS360_Project/Dataset/coloured/ab2.npy') #second 10k coloured images
colour3 = np.load('/content/drive/My Drive/APS360_Project/Dataset/coloured/ab3.npy') #last 5k coloured images


#the number of images in each file
print(len(gray), len(colour1), len(colour2), len(colour3))


