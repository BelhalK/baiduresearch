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