import json 

#create a dictionary object
person_dict = { 'first': 'Christopher', 'last':'Harrison'}

#Add an item 
person_dict['City']='Seattle'

print(person_dict)

#Convert Dictionary to JSON object
person_json = json.dumps(person_dict)
print(person_json)
