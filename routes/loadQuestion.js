
var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://localhost:27017/';
let assignQuestions = require('./assignQuestions')

Base = 50
noiseLevels = [128,64,32,16,8,4,2,1]

const loadModule = { 
    
    loadFirst: function(req, res, user) {

        //determine noise level from position of id
        noiselevel = noiseLevels[0];

        var question2load
        var questionArray

        co(function* () {

            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsbasic')
            let usersCol = db.collection('users')
            let questionPoolCol = db.collection('questionPool')
        
            let assignedQuestions = assignQuestions.assign();
        
            check = yield usersCol.findOne({"user" : user.id})
        
            //check to see if user exists in database
            if(check === null && user.id != null){
              
              //insert new user if user does not exist
                var item = { 
                    "user": user.id,
                    "key2pay": null,
                    "surveyResults": null,
                    "group4Answers": assignedQuestions
                };
        
                yield usersCol.insertOne(item);
        
               //load next question

                //find question pool for user
                var questions =  yield usersCol.find({"user": user.id}).toArray();       
                questions = questions[0].group4Answers

                // get question array instance at the position of id
                let variation = questions[0];

                //find question from pool based off of the noise level and variation
                question2load = yield questionPoolCol.find({"noiselevel": noiselevel, "variation": variation}).toArray();
                question2load = question2load[0].array

                question = JSON.stringify(question2load)

                //console.log("question: " + question)

                user.saveCurrentQuestion(question)

                res.render('rankings', { userID: user.id , id: user.activityID , type: "rankings", question: user.question() , noiselevel})
            } else{
                res.render('index', {error: "ERROR: Username already exists"});
            }
        })
    },

    loadAfterRanking: function(req, res, user) {

      console.log(user.id)
      console.log(user.activityID)
      
      //determine noise level from position of id
      noiselevel = noiseLevels[user.activityID-1];
      var question2load;

      co(function* () {

        let client = yield MongoClient.connect(url);
        const db = client.db('ratingsrankingsbasic')
        let usersCol = db.collection('users')
        let responseCol = db.collection('responses')
        let questionPoolCol = db.collection('questionPool')

        console.log("activityID: ", user.activityID)
        console.log("user: ", user.id)

        check =  yield responseCol.findOne({"user": user.id, "collection": String(user.activityID), "type": 'ranking'})

        console.log("check: ", check)

        
        if (check === null){
          res.render('rankings', {userID : user.id, id: user.activityID , type: "rankings", question: user.question(), noiselevel, error: "ERROR: Please submit a complete ranking"})
          return;
        } 
        
        res.render('ratings', {userID: user.id, id: user.activityID, type: "ratings", picture: 0, question: user.question(), noiselevel});

      });

    },

    loadAfterRating: function(req, res, user, picture){

      noiselevel = noiseLevels[user.activityID - 1];
      var question2load

      if(parseInt(picture) === 3){
        picture === 0

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db('ratingsrankingsbasic')
          let usersCol = db.collection('users')
          let questionPoolCol = db.collection('questionPool')
  
          //find users questions
          var questions =  yield usersCol.find({"user": user.id }).toArray();       
          questions = questions[0].group4Answers
  
          // get question array instance at the position of id
          let variation = questions[id-1];
  
          //find question from pool based off of the noise level and variation
          question2load = yield questionPoolCol.find({"noiselevel": noiselevel, "variation": variation}).toArray();
          question2load = question2load[0].array
          //console.log("questions2load ", question2load)
  
          question = JSON.stringify(question2load)

          user.saveCurrentQuestion(question)
          
          //adjust to next activity

          res.render('rankings', {userID: user.id, id: user.activityID , type: "rankings", question: user.question(), noiselevel})
      
        });
      } else {
        picture = parseInt(picture)+ 1
        res.render('ratings', {userID: user.id, id: user.activityID, type: "ratings", picture, question: user.question(), noiselevel})
      }

    }
}

module.exports = loadModule
