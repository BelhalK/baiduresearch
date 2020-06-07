def accuracies(adagrad, adabound, adam, padam, rmsprop, sgd, sagd):
  adagradacc = adagrad['test_acc']
  adaboundacc = adabound['test_acc']
  adamacc = adam['test_acc']
  padamacc = padam['test_acc']
  rmspropacc = rmsprop['test_acc']
  sgdacc = sgd['test_acc']
  sagdacc = sagd['test_acc']
  list_maxs = []
  for element in adagradacc:
      mins = []
      for ind in element:
          mins.append(max(ind))
      list_maxs.append(mins)
  adagradacc2 = list(map(list, zip(*list_maxs)))
  list_maxs = []
  for element in adaboundacc:
      mins = []
      for ind in element:
          mins.append(max(ind))
      list_maxs.append(mins)
  adaboundacc2 = list(map(list, zip(*list_maxs)))
  list_maxs = []
  for element in adamacc:
      mins = []
      for ind in element:
          mins.append(max(ind))
      list_maxs.append(mins)
  adamacc2 = list(map(list, zip(*list_maxs)))
  list_maxs = []
  for element in padamacc:
      mins = []
      for ind in element:
          mins.append(max(ind))
      list_maxs.append(mins)
  padamacc2 = list(map(list, zip(*list_maxs)))
  list_maxs = []
  for element in rmspropacc:
      mins = []
      for ind in element:
          mins.append(max(ind))
      list_maxs.append(mins)
  rmspropacc2 = list(map(list, zip(*list_maxs)))
  list_maxs = []
  for element in sgdacc:
      mins = []
      for ind in element:
          mins.append(max(ind))
      list_maxs.append(mins)
  sgdacc2 = list(map(list, zip(*list_maxs)))
  list_maxs = []
  for element in sagdacc:
      mins = []
      for ind in element:
          mins.append(max(ind))
      list_maxs.append(mins)
  sagdacc2 = list(map(list, zip(*list_maxs)))
  return adagradacc2, adaboundacc2, adamacc2, padamacc2, rmspropacc2, sgdacc2, sagdacc2

def losses(adagradloss, adaboundloss, adamloss, padamloss, rmsproploss, sgdloss, sagdloss):
  list_mins = []
  for element in sagdloss:
      mins = []
      for ind in element:
          mins.append(min(ind))
      list_mins.append(mins)
  sagdloss2 = list(map(list, zip(*list_mins)))
  list_mins = []
  for element in adagradloss:
      mins = []
      for ind in element:
          mins.append(min(ind))
      list_mins.append(mins)
  adagradloss2 = list(map(list, zip(*list_mins)))
  list_mins = []
  for element in adaboundloss:
      mins = []
      for ind in element:
          mins.append(min(ind))
      list_mins.append(mins)
  adaboundloss2 = list(map(list, zip(*list_mins)))
  list_mins = []
  for element in adamloss:
      mins = []
      for ind in element:
          mins.append(min(ind))
      list_mins.append(mins)
  adamloss2 = list(map(list, zip(*list_mins)))
  list_mins = []
  for element in padamloss:
      mins = []
      for ind in element:
          mins.append(min(ind))
      list_mins.append(mins)
  padamloss2 = list(map(list, zip(*list_mins)))
  list_mins = []
  for element in rmsproploss:
      mins = []
      for ind in element:
          mins.append(min(ind))
      list_mins.append(mins)
  rmsproploss2 = list(map(list, zip(*list_mins)))
  list_mins = []
  for element in sgdloss:
      mins = []
      for ind in element:
          mins.append(min(ind))
      list_mins.append(mins)
  sgdloss2 = list(map(list, zip(*list_mins)))
  return adagradloss2, adaboundloss2, adamloss2, padamloss2, rmsproploss2, sgdloss2, sagdloss2