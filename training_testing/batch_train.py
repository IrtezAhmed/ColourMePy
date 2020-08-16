# this training procedure modularizes the training process and reduces the required memory of your training process
# this is ideal for computers with low RAM
#simply running this code will begin training


active = True
num_trial = 0
actualEpoch = 0
canLoad=False


goodTry = []
while active: #low memory training over 10000 images

  x,y, train_loader,ground_loader = 0,0,0,0

  print("Next")

  if num_trial == 0:
    x, y = createData()
  elif num_trial == 1:
    x, y = createData()
  elif num_trial == 2:
    x, y = createData2()
  elif num_trial == 3:
    x, y = createData3()
  elif num_trial == 4:
    x, y = createData4()
  elif num_trial == 5:
    x, y = createData5()
  elif num_trial == 6:
    x, y = createData6()
  elif num_trial == 7:
    x, y = createData7()
  elif num_trial == 8:
    x, y = createData8()
  elif num_trial == 9:
    x, y = createData9()
    actualEpoch+=1
    num_trial=-1
    
  
  if gemini:
    break

  for i, b in zip(x, y):
    train_loader = torch.utils.data.DataLoader(b)
    ground_loader = torch.utils.data.DataLoader(i)
    result = train(IrtezaNet())

  num_trial+=1
  print("Trial",num_trial,"complete!")

  if len(goodTry)>=270:
    what = input()

  if actualEpoch==5:
    active = False