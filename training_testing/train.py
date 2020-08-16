criterion = nn.MSELoss()

def train(model, num_epochs=5): 
    torch.manual_seed(1000) 
    optimizer = torch.optim.Adam(model.parameters()) 

    epoch_load=0
    results = [] #creates list

    print("Training in progress...")

    global canLoad

    if canLoad:
      state = torch.load(save_path)

      model.load_state_dict(state['state_dict'])
      optimizer.load_state_dict(state['optimizer'])
      epoch_load = state['epoch']
    else:
      canLoad=True


    for epoch in range(num_epochs):
        for data, truth in zip(train_loader, ground_loader): 

          img = data.float()
          groundTruth = truth.float()
          output = model(img) 
          loss = criterion(output, groundTruth) 

          if float(loss)<50: 
            goodTry.append(output)
            goodTry.append(groundTruth)
            goodTry.append(img)

          if float(loss)<30:
            gemini = True

          loss.backward() 
          optimizer.step() 
          optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+epoch_load+1, float(loss)))

        state = {
          'epoch': epoch+epoch_load+1 , 'state_dict': model.state_dict() , 'optimizer': optimizer.state_dict()
      } 
        torch.save(state, save_path)

    return results


    