var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';
//var url = 'mongodb://localhost:27017/';

let assignQuestions = require('./assignBatch')

const storeModule = {

    storeRanking: function(userID, id, group2save, time, frames, batch){

        //store into db
        co(function* () {

            var group = group2save.map(Number);
            console.log(group)
            
            //connect to db
            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsframes')
            let responseCol = db.collection('responses')

            var item = {
                "user" : userID,
                "collection": id,
                "frames": frames,
                "batch": batch,
                "type": "ranking",
                "ranking": group2save,
                "time": time
            }

            var criteria = {
                "user": userID, 
                "collection": id, 
                "type": "ranking"
            }

            var newItem = {
                "ranking": group2save,
                "time": time
            }

            count = yield responseCol.find(criteria).count()

            if(count > 0){
                responseCol.update(criteria,{ $set: newItem })
                //console.log('Ranking updated')
            } else {
                responseCol.insertOne(item, function(err, result) {
                //console.log('Ranking inserted')
                });
        }

        client.close();
      
        });

    },

    storeRating: function(userID, id, picture, rating, time, batch, frames) {

        //insert rating into db
        co(function* () {

            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsframes')
            let responseCol = db.collection('responses')

            var item = {
                "user" : userID,
                "collection": id,
                "batch": batch,
                "frames": frames,
                "type": "rating",
                "picture": picture,
                "estimate": rating,
                "time": time
            }

            responseCol.insertOne(item, function(err, result) {
                //console.log('Rating inserted')
            });
        });
    },

    storeSurvey: function(userID, result, key){

        co(function* () {
            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsframes')
            let UsersCol = db.collection('users')

            newItem = {
                "surveyResults": result, 
                "key2pay": key
            }

            UsersCol.updateOne({"user": userID}, { $set: newItem });
            //console.log('User Completed task')
        })
    }
}

module.exports = storeModule
