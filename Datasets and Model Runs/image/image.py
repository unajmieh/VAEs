from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
cloud = fetch_ucirepo(id=155) 
  
# data (as pandas dataframes) 
X = cloud.data.features 
y = cloud.data.targets 
  
# metadata 
print(cloud.metadata) 
  
# variable information 
print(cloud.variables) 
