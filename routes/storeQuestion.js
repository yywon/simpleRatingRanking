var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

//var url = 'mongodb://localhost:27017/';
var url = 'mongodb://10.218.105.218:27017/';
let assignQuestions = require('./assignQuestions')

const storeModule = {

    storeRanking: function(userID, id, group2save, time){

        //store into db
        co(function* () {

            var group = group2save.map(Number);
            console.log(group)

            //sort array
            group.sort((a,b) => a - b);
            
            //calc sum difference 
            n = group.length
            diffSum = 0;
            for(i = n - 1; i >= 0; i--){
                diffSum += i*group[i] - (n-1-i) * group[i]
            }
            diffSum = Math.abs(diffSum)
            

            //min amongst all pairs
            let pairs = []
            for (let i = 0; i < group.length - 1; i++) {
                for (let j = i + 1; j < group.length; j++) {
                    p = [group[i], group[j]]
                    pairs.push(p)
                }
            }

            min = 99
            for(i = 0; i < pairs.length - 1; i++){
                test = Math.abs(pairs[i][0]- pairs[i][1])
                if(test < min){
                    min = test
                }
            } 

            console.log("diffSum: ", diffSum)
            console.log("min", min)

            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsdistributed')
            let responseCol = db.collection('responses')

            var item = {
                "user" : userID,
                "collection": id,
                "type": "ranking",
                "ranking": group2save,
                "sum_differences": diffSum,
                "minimum_difference": min, 
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
                console.log('Ranking updated')
                console.log('total time', time)
            } else {
                responseCol.insertOne(item, function(err, result) {
                console.log('Ranking inserted')
                });
        }

        client.close();
      
        });

    },

    storeRating: function(userID, id, picture, rating, time) {

        //insert rating into db
        co(function* () {

            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsdistributed')
            let responseCol = db.collection('responses')

            var item = {
                "user" : userID,
                "collection": id,
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
            const db = client.db('ratingsrankingsdistributed')
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
