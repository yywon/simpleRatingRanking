var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';
let assignQuestions = require('./assignQuestions')

const loadModule = { 
    
    loadFirst: function(req, res, user) {

      let framesList = [2,3,5,6]

        var question2load
        var questionArray

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db('ratingsrankingsdistributed')
          let usersCol = db.collection('users')
          let responseCol = db.collection('responses')

            i = Math.floor(Math.random() * 4) 
            let userFrames = framesList[i] 
            let assignedQuestions = assignQuestions.assign(userFrames)

            console.log(assignedQuestions)

            check = yield usersCol.findOne({"user" : user.id})

            //check to see if user exists in database
            if(check === null && user.id != null){
              
              //insert new user if user does not exist
                var item = { 
                    "user": user.id,
                    "key2pay": null,
                    "surveyResults": null,
                    "group4Answers": assignedQuestions,
                    "frames": userFrames
                };
                
                yield usersCol.insertOne(item);
                
               //load next question

                //find question pool for user
                var questions =  yield usersCol.find({"user": user.id}).toArray();       
                questions = questions[0].group4Answers

                // get question array instance at the position of id
                let question2load = questions[0];
                
                question = JSON.stringify(question2load)
                console.log("question: " + question)
                user.saveCurrentQuestion(question)
                console.log("user frames: " + userFrames)
                user.saveFrames(userFrames)

                res.render('rankings', { userID: user.id , id: user.activityID , type: "rankings", frames: user.getFrames(), total: user.getTotal(), question: user.question()})
            } else{
                res.render('index', {error: "ERROR: Username already exists"});
            }
        })
    },

    loadAfterRanking: function(req, res, user) {

      co(function* () {

        let client = yield MongoClient.connect(url);
        const db = client.db('ratingsrankingsdistributed')
        let usersCol = db.collection('users')
        let responseCol = db.collection('responses')

        check =  yield responseCol.findOne({"user": user.id, "collection": String(user.activityID), "type": 'ranking'})

        if (check === null){
          res.render('rankings', {userID : user.id, id: user.activityID , type: "rankings", question: user.question(), total: user.getTotal(), frames: user.getFrames(), error: "ERROR: Please submit a complete ranking"})
          return;
        } 
        
        res.render('ratings', {userID: user.id, id: user.activityID, type: "ratings", picture: 0, total: user.getTotal(), question: user.question()});

      });

    },

    loadAfterRating: function(req, res, user, picture){

      var question2load

      if(parseInt(picture) === user.getFrames() - 1){
        picture === 0

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db('ratingsrankingsdistributed')
          let usersCol = db.collection('users')
          let responseCol = db.collection('responses')

          //find users questions
          var questions =  yield usersCol.find({"user": user.id }).toArray();       
          questions = questions[0].group4Answers
  
          let question2load = questions[user.activityID - 1];
                
          question = JSON.stringify(question2load)

          console.log("question: " + question)

          user.saveCurrentQuestion(question)
          
          //adjust to next activity

          res.render('rankings', {userID: user.id, id: user.activityID , frames: user.getFrames(), total: user.getTotal(), type: "rankings", question: user.question()})
      
        });
      } else {
        picture = parseInt(picture)+ 1
        res.render('ratings', {userID: user.id, id: user.activityID, type: "ratings", picture, total: user.getTotal(), question: user.question()})
      }

    }
}

module.exports = loadModule
