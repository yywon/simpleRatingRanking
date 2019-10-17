
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
    
    // loadFirst: function(req, res, userID, id) {
    //NOTE: id is now available via user.activityID
    loadFirst: function(req, res, user) {

        //TODO: use user.id, user.activityID where necessary

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
        
                console.log(usersCol.insertOne);
        
                yield usersCol.insertOne(item);
        
               //load next question

                //console.log("userID", userID)

                //find question pool for user
                var questions =  yield usersCol.find({"user": user.id}).toArray();       
                questions = questions[0].group4Answers

                // get question array instance at the position of id
                let variation = questions[0];

                //find question from pool based off of the noise level and variation
                question2load = yield questionPoolCol.find({"noiselevel": noiselevel, "variation": variation}).toArray();
                //console.log("questions2load ", question2load)
                question2load = question2load[0].array
                //console.log("questions2load ", question2load)

                question = JSON.stringify(question2load)

                //console.log("question: " + question)

                //TODO: before sending question, save question to user's instance via user.saveCurrentQuestion
                user.saveCurrentQuestion(question)

                res.render('rankings', { user: user.id , id: user.activityID , type: "rankings", question: user.question() , noiselevel})
            } else{
                res.render('index', {error: "ERROR: Username already exists"});
            }
        })
    },

    loadAfterRanking: function(req, res, user) {

      userID = user.id
      id = user.activityID
      
      //determine noise level from position of id
      noiselevel = noiseLevels[id-1];
      var question2load;

      co(function* () {

        let client = yield MongoClient.connect(url);
        const db = client.db('ratingsrankingsbasic')
        let usersCol = db.collection('users')
        let responseCol = db.collection('responses')
        let questionPoolCol = db.collection('questionPool')

        //check if inserted
        check =  yield responseCol.findOne({"user": user.id, "type":"ranking", "collection": user.activityID})

        //TODO: Change question

        if (check === null){
          res.render('rankings', {userID : user.ID, id: user.activityID , type: "rankings", question: user.question(), noiselevel, error: "ERROR: Please submit a complete ranking"})
          return;
        } 

        //find users questions
        var questions =  yield usersCol.find({"user":userID}).toArray();       
        questions = questions[0].group4Answers
        //console.log("questions ", questions)

        // get question array instance at the position of id
        let variation = questions[id-1];

        //find question from pool based off of the noise level and variation
        //question2load = yield questionPoolCol.find( {"questions" : {"noiselevel": noiselevel, "variation": variation} } ).toArray();
        question2load = yield questionPoolCol.find({"noiselevel": noiselevel, "variation": variation}).toArray();
        question2load = question2load[0].array
        //console.log("questions2load ", question2load)

        //  TODO: update user's current question here
        question = JSON.stringify(question2load)
        user.saveCurrentQuestion(question)

        res.render('ratings', {userID, id, type: "ratings", picture: 0, question, noiselevel});

      });

    },

    loadAfterRating: function(req, res, user, picture){

      userID = user.id
      id = user.activityID

      noiselevel = noiseLevels[user.activityID - 1];
      var question2load

      //console.log("userID: ", userID)
      //console.log("id: ", id)

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

          res.render('rankings', {userID: user.id, id: user.activityID , type: "rankings", question: user.question, noiselevel})
      
        });
      } else {
        picture = parseInt(picture)+ 1
        res.render('ratings', {userID: user.id, id: user.activityID, type: "ratings", picture, question: user.question, noiselevel})
      }

    }
}

module.exports = loadModule
