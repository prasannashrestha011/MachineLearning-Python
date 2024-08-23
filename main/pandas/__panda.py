import pandas as pd
data={
    "name":["shyam","ram","hari"],
    "age":[18,19,20],
    "address":["bhaktapur","kathamandu","palpa"],
    
}
panda_data=pd.DataFrame(data)
print(panda_data,"\n")
## for displaying data with condition
filtered_data=panda_data[panda_data.age>18]
print(filtered_data)
