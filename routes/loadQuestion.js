var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

//var url = 'mongodb://10.218.105.218:27017/';
var url = 'mongodb://localhost:27017/';

var dbase = 'ratingsrankingsB1'

const User = require('../User');

const loadModule = { 
    
    loadFirst: function(req, res, user, userOrder) {

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db(dbase)
          let usersCol = db.collection('users')
          let batchesCol = db.collection('batches')

          assignedQuestions = []
          assignedIndexes = []

          //get assigned questions for B
          for(i = 0; i < userOrder.length; i++){

            frame = userOrder[i]
            console.log(frame)

            findQuestions2:
            for(batch = 0; batch < 180/(60/frame); batch++){

              console.log(batch)

              dbBatch = yield batchesCol.findOne({'size': frame, 'number': batch})
            
              for(question = 0; question < dbBatch.assignmentStatus.length; question++){
                if(dbBatch.assignmentStatus[question] === 0){

                  console.log('question, ', question)
    
                  assignedQuestions.push(dbBatch.questions[question])
                  assignedIndexes.push([frame, dbBatch.number, question])  

                  update = {"$set": {}}
                  update['$set']["assignmentStatus."+question] = 1

                  batchesCol.updateOne( {'size': dbBatch.size, 'number': dbBatch.number}, update)

                  break findQuestions2;
                }
              }
            }
          }

          //assign ab order to user
          userNumber = yield usersCol.count()

          console.log("assigned Quesions: ", assignedQuestions)
          console.log("assigned Indexes: ", assignedIndexes)

          user.saveQuestionOrder(assignedQuestions)
          user.saveIndexOrder(assignedIndexes)

          check = yield usersCol.findOne({"user" : user.id})
              
          //check to see if user exists in database
          if(check === null && user.id != null){
            
            //insert new user if user does not exist
              var item = { 
                  "user": user.id,
                  "key2pay": null,
                  "surveyResults": null,
                  "questions": user.getQuestionOrder(),
                  "indexes": user.getIndexOrder(),
              };
              
              yield usersCol.insertOne(item);
          
              //load first question
              questionOrder = user.getQuestionOrder()
              question2load = questionOrder[0]
              questionLength = question2load.length
              indexOrder = user.getIndexOrder()
              questionIndex = indexOrder[0]
              currentBatch = questionIndex[1]
              user.saveCurrentQuestion(JSON.stringify(question2load), currentBatch, questionLength)
            
              res.render('rankings', { userID: user.id , id: user.activityID , type: "rankings", frames: user.frames(), question: user.question()})

          } else{
              res.render('index', {error: "ERROR: Username already exists"});
          }
      })
    },

    loadAfterRanking: function(req,res,user) {
      co(function* () {
        
        //connect to db
        let client = yield MongoClient.connect(url);
        const db = client.db('ratingsrankingsB')
        let usersCol = db.collection(dbase)
        let responseCol = db.collection('responses')

        check =  yield responseCol.findOne({"user": user.id, "collection": String(user.activityID), "type": 'ranking'})

        if (check === null){
          res.render('rankings', {userID : user.id, id: user.activityID , type: "rankings", frames: user.frames(), question: user.question(), error: "ERROR: Please submit a complete ranking"})
          return;
        } else{
          OrderRanked = JSON.stringify(check.ranking)
          res.render('ratings2', {userID: user.id, id: user.activityID, type: "ratings", frames: user.frames(), question: OrderRanked});
        }
      })
    },

    loadAfterRating: function(req, res, user){

      console.log('in the func')
      console.log(user)

      //load the next rating question

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db(dbase)
          let usersCol = db.collection('users')
          let responseCol = db.collection('responses')
          
          //check if response made it to db
          check =  yield responseCol.findOne({"user": user.id, "collection": String(user.activityID), "type": 'rating'})
          console.log('check')

          if (check === null){
            res.render('ratings2', {userID : user.id, id: user.activityID , type: "ratings", frames: user.frames(), question: user.question(), error: "ERROR: Please submit valid estimates"})
            return;
          } else{

            user.activityID += 1
            user.studyQuestion += 1

            if(user.studyQuestion === 5){
              res.render('survey', {userID: user.id})
            } else {

              //load next question
              questionOrder = user.getQuestionOrder()
              console.log(questionOrder)
              question2load = questionOrder[user.studyQuestion - 1]
              console.log(question2load)
              questionLength = question2load.length

              indexOrder = user.getIndexOrder()
              questionIndex = indexOrder[user.studyQuestion - 1]
              currentBatch = questionIndex[1]
              user.saveCurrentQuestion(JSON.stringify(question2load), currentBatch, questionLength)
              
              //adjust to next activity

              res.render('rankings', {userID: user.id, id: user.activityID , frames: user.frames(), type: "rankings", question: user.question()})
            }
          }
        });
    }, 

    loadSurvey: function(req,res,user){
      res.render('survey', {user: user.id})
    }

}

module.exports = loadModule
