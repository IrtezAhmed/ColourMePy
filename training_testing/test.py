def test(model): #test model on data with updating parameters
    torch.manual_seed(1000) 
    criterion = nn.MSELoss() 

    results = [] 

    print("TESTING")

    state = torch.load(save_path)

    model.load_state_dict(state['state_dict'])

    for data, truth in zip(test_loader, truth_loader): 

      img = data.float()
      groundTruth = truth.float()
      output = model(img) 
      loss = criterion(output, groundTruth)  
      print(loss.float())
      results.append(output) 
      results.append(groundTruth)
      results.append(groundTruth)

    return results


#load testing data, values can be changed as liked

test_truth = combine(gray[15010:], colour2[5010:], 10)
test_input = grayify(gray[15010:], 10)

test_loader = torch.utils.data.DataLoader(test_input)
truth_loader = torch.utils.data.DataLoader(test_truth)

showAllInput(test_truth)

