import json 

#create a dictionary object
person_dict = { 'first': 'Christopher', 'last':'Harrison'}

#Add an item 
person_dict['City']='Seattle'

print(person_dict)

#Convert Dictionary to JSON object
person_json = json.dumps(person_dict)
print(person_json)

#Create staff dictionary 
staff_dict = {}
staff_dict ['Program Manager'] = person_dict
staff_json = json.dumps(staff_dict)

print(staff_json)

# List object of dictionaries

languages_list = ['CSharp', 'Python', 'JavaScript']
person_dict['languages'] = languages_list

#Convert dictionary to JSON object
person_json = json.dumps(person_dict)
print(person_json)
