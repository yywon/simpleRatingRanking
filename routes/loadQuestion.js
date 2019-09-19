var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';

Base = 50
noiseLevels = [1,2,4,8,16,32,64,128]

const loadModule = { 

    load: function(userID, id) {

        id = id-1
        //determine noise level from position of id
        noiselevel = noiseLevels[id];

        var question2load

        co(function*(){

            //set up db and collections
            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsbasic')
            let usersCol = db.collection('users')
            let questionPoolCol = db.collection('questionPool')

            console.log("userID", userID)

            //find question pool for user
            var questions =  yield usersCol.find({"user":userID}).toArray();       
            questions = questions[0].group4Answers
            console.log("questions ", questions)

            // get question array instance at the position of id
            let variation = questions[id];

            console.log("noiselevel: ", noiselevel)
            console.log("variation: ", variation)

            //find question from pool based off of the noise level and variation
            question2load = yield questionPoolCol.find( {"questions" : {"noiselevel": noiselevel, "variation": variation} } ).toArray();
            question2load = question2load[0].array
            console.log("questions2load ", question2load)


        })

        //questionArray = [noiselevel, question2load]

        questionArray = [32, [146, 50, 82, 114]]

        console.log("question array: " + questionArray)

        return questionArray

    }
}

module.exports = loadModule
