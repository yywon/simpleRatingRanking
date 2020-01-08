var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';
//var url = 'mongodb://localhost:27017/';

const User = require('../User');

const loadModule = { 
    
    loadFirst: function(req, res, user, userOrder) {

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db('ratingsrankingsframes')
          let usersCol = db.collection('users')
          let batchesCol = db.collection('batches')

          assignedQuestions = []
          assignedIndexes = []

          for(i = 0; i < userOrder.length; i++){

            frame = userOrder[i]
            console.log(frame)

            findQuestions:
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

                  break findQuestions;
                }
              }
            }
          }
          
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
                  "indexes": user.getIndexOrder()
              };
              
              yield usersCol.insertOne(item);
              
              //load first question
              questionOrder = user.getQuestionOrder()
              question2load = questionOrder[0]
              questionLength = question2load.length
              indexOrder = user.getIndexOrder()
              questionIndex = indexOrder[0]
              currentBatch = questionIndex[0]
              user.saveCurrentQuestion(JSON.stringify(question2load), currentBatch, questionLength)
              
              res.render('rankings', { userID: user.id , id: user.activityID , type: "rankings", frames: user.frames(), question: user.question()})
          } else{
              res.render('index', {error: "ERROR: Username already exists"});
          }
      })
    },

    loadAfterRanking: function(req, res, user) {

      co(function* () {

        let client = yield MongoClient.connect(url);
        const db = client.db('ratingsrankingsframes')
        let usersCol = db.collection('users')
        let responseCol = db.collection('responses')

        check =  yield responseCol.findOne({"user": user.id, "collection": String(user.activityID), "type": 'ranking'})

        if (check === null){
          res.render('rankings', {userID : user.id, id: user.activityID , type: "rankings", frames: user.frames(), question: user.question(), error: "ERROR: Please submit a complete ranking"})
          return;
        } else{
          res.render('ratings', {userID: user.id, id: user.activityID, type: "ratings", picture: 0, question: user.question()});
        }

      });

    },

    loadAfterRating: function(req, res, user, picture){

      if(parseInt(picture) === user.frames() - 1){
        picture === 0

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db('ratingsrankingsframes')
          let usersCol = db.collection('users')
          let responseCol = db.collection('responses')



          //load next question
          questionOrder = user.getQuestionOrder()
          question2load = questionOrder[user.activityID - 1]
          questionLength = question2load.length

          indexOrder = user.getIndexOrder()
          questionIndex = indexOrder[0]
          currentBatch = questionIndex[0]
          user.saveCurrentQuestion(JSON.stringify(question2load), currentBatch, questionLength)
          
          //adjust to next activity

          res.render('rankings', {userID: user.id, id: user.activityID , frames: user.frames(), type: "rankings", question: user.question()})
      
        });

      } else {
        picture = parseInt(picture)+ 1
        res.render('ratings', {userID: user.id, id: user.activityID, type: "ratings", picture, question: user.question()})
      }

    }
}

module.exports = loadModule
