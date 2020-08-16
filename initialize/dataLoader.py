#functions to create data

def createData():

  all_ground = []
  all_input = []
   

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground.append(combineAll(gray, colour1, 0+c))
    all_input.append(grayifyAll(gray, 0+c))
  
  return all_ground, all_input


def createData1():
  
  all_ground1 = []
  all_input1 = []

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground1.append(combineAll(gray, colour1, 100+c))
    all_input1.append(grayifyAll(gray, 0+c))

  return all_ground1, all_input1

def createData2():

  all_ground2 = []
  all_input2 = []

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground2.append(combineAll(gray, colour1, 200+c))
    all_input2.append(grayifyAll(gray, 0+c))

  return all_ground2, all_input2


def createData3():

  all_ground3 = []
  all_input3 = []
  

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground3.append(combineAll(gray, colour1, 300+c))
    all_input3.append(grayifyAll(gray, 0+c))

  return all_ground3, all_input3

def createData4():

  all_ground4 = []
  all_input4 = []
  

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground4.append(combineAll(gray, colour1, 400+c))
    all_input4.append(grayifyAll(gray, 0+c))

  return all_ground4, all_input4


def createData5():

  all_ground5 = []
  all_input5 = []
  

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground5.append(combineAll(gray, colour1, 500+c))
    all_input5.append(grayifyAll(gray, 0+c))

  return all_ground5, all_input5


def createData6():

  all_ground6 = []
  all_input6 = []
  

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground6.append(combineAll(gray, colour1, 600+c))
    all_input6.append(grayifyAll(gray, 0+c))

  return all_ground6, all_input6


def createData7():

  all_ground7 = []
  all_input7 = []

  

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground7.append(combineAll(gray, colour1, 700+c))
    all_input7.append(grayifyAll(gray, 0+c))

  return all_ground7, all_input7


def createData8():

  all_ground8 = []
  all_input8 = []
  
  for i in range(iterations):
      b = all_batch_num*i
      c = b-all_batch_num
      all_ground8.append(combineAll(gray, colour1, 800+c))
      all_input8.append(grayifyAll(gray, 0+c))

  return all_ground8, all_input8


def createData9():
  
  all_ground9 = []
  all_input9 = [] 

  for i in range(iterations):
    b = all_batch_num*i
    c = b-all_batch_num
    all_ground9.append(combineAll(gray, colour1, 900+c))
    all_input9.append(grayifyAll(gray, 0+c))

  return all_ground9, all_input9