from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')

# data base name : 'test-database-1'
mydb = client['test-database-1']

import datetime

myrecord = {
        "author": "Duke",
        "title" : "PyMongo 101",
        "tags" : ["MongoDB", "PyMongo", "Tutorial"],
        "date" : datetime.datetime.utcnow()
        }

record_id = mydb.mytable.insert(myrecord)

print (record_id)
print (mydb.collection_names())
collection=mydb['mytable']
cursor = collection.find({})
for document in cursor:
    print(document)

myrecord2 = [
        { "author": "Duke II",
          "title" : "PyMongo II 101",
          "tags" : ["MongoDB II", "PyMongo II", "Tutorial II"],
          "date" : datetime.datetime.utcnow() },
        { "author": "Duke III",
          "title" : "PyMongo III 101",
          "tags" : ["MongoDB III", "PyMongo III", "Tutorial III"],
          "date" : datetime.datetime.utcnow() }
        ]

mydb.mytable.insert(myrecord2)

print(mydb.collection_names())

collection=mydb['mytable']
cursor = collection.find({})
for document in cursor:
    print(document)
