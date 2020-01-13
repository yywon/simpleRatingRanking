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
          let batchesColA = db.collection('batchesA')
          let batchesColB = db.collection('batchesB')


          assignedQuestionsA = []
          assignedIndexesA = []

          assignedQuestionsB = []
          assignedIndexesB = []


          //NOTE: There might be a bug here

          //get assigned questions for A
          for(i = 0; i < userOrder.length; i++){

            frame = userOrder[i]
            console.log(frame)

            findQuestions:
            for(batch = 0; batch < 180/(60/frame); batch++){

              console.log(batch)

              dbBatch = yield batchesColA.findOne({'size': frame, 'number': batch})
            
              for(question = 0; question < dbBatch.assignmentStatus.length; question++){
                if(dbBatch.assignmentStatus[question] === 0){

                  console.log('question, ', question)
    
                  assignedQuestionsA.push(dbBatch.questions[question])
                  assignedIndexesA.push([frame, dbBatch.number, question])  

                  update = {"$set": {}}
                  update['$set']["assignmentStatus."+question] = 1

                  batchesColA.updateOne( {'size': dbBatch.size, 'number': dbBatch.number}, update)

                  break findQuestions;
                }
              }
            }
          }

          //get assigned questions for B
          for(i = 0; i < userOrder.length; i++){

            frame = userOrder[i]
            console.log(frame)

            findQuestions:
            for(batch = 0; batch < 180/(60/frame); batch++){

              console.log(batch)

              dbBatch = yield batchesColB.findOne({'size': frame, 'number': batch})
            
              for(question = 0; question < dbBatch.assignmentStatus.length; question++){
                if(dbBatch.assignmentStatus[question] === 0){

                  console.log('question, ', question)
    
                  assignedQuestionsB.push(dbBatch.questions[question])
                  assignedIndexesB.push([frame, dbBatch.number, question])  

                  update = {"$set": {}}
                  update['$set']["assignmentStatus."+question] = 1

                  batchesColB.updateOne( {'size': dbBatch.size, 'number': dbBatch.number}, update)

                  break findQuestions;
                }
              }
            }
          }

          //assign ab order to user
          userNumber = yield usersCol.count()

          if(userNumber % 2 === 0){
            user.saveABOrder("ab")
          } else{
            user.saveABOrder("ba")
          }
          
          console.log("assigned Quesions: ", assignedQuestions)
          console.log("assigned Indexes: ", assignedIndexes)

          user.saveQuestionOrderA(assignedQuestionsA)
          user.saveIndexOrderA(assignedIndexesA)

          user.saveQuestionOrderB(assignedQuestionsA)
          user.saveIndexOrderB(assignedIndexesA)

          check = yield usersCol.findOne({"user" : user.id})
              
          //check to see if user exists in database
          if(check === null && user.id != null){
            
            //insert new user if user does not exist
              var item = { 
                  "user": user.id,
                  "key2pay": null,
                  "surveyResults": null,
                  "ABorder": user.getABOrder(),
                  "questionsA": user.getQuestionOrderA(),
                  "indexesA": user.getIndexOrderA(),
                  "questionsB": user.getQuestionOrderB(),
                  "indexesB": user.getIndexOrderB(),

              };
              
              yield usersCol.insertOne(item);
              

              //condition if user is ab
              if(user.getABOrder() === "ab"){
                //load first question
                questionOrder = user.getQuestionOrderA()
                question2load = questionOrder[0]
                questionLength = question2load.length
                indexOrder = user.getIndexOrderA()
                questionIndex = indexOrder[0]
                currentBatch = questionIndex[0]


                user.saveCurrentQuestion("a", JSON.stringify(question2load), currentBatch, questionLength)
              
                res.render('rankings', { userID: user.id , id: user.activityID , type: "rankings", frames: user.frames(), question: user.question()})

                //condition if user is ba
              } else {

                //load first question
                questionOrder = user.getQuestionOrderB()
                question2load = questionOrder[0]
                questionLength = question2load.length
                indexOrder = user.getIndexOrderA()
                questionIndex = indexOrder[0]
                currentBatch = questionIndex[0]


                user.saveCurrentQuestion("b", JSON.stringify(question2load), currentBatch, questionLength)
              
                res.render('rankings2', { userID: user.id , id: user.activityID , type: "rankings", frames: user.frames(), question: user.question()})

              }

          } else{
              res.render('index', {error: "ERROR: Username already exists"});
          }
      })
    },

    loadNextStudy: function(req, res, user){


      //Option 1:
      res.render('survey', {userID: currentUser.id})

      //Option 2:
      res.render('rankings', { userID: user.id , id: user.activityID , type: "rankings", frames: user.frames(), question: user.question()})

      //Option 3:
      res.render('rankings2', { userID: user.id , id: user.activityID , type: "rankings", frames: user.frames(), question: user.question()})


    }, 

    loadAfterRankingA: function(req, res, user) {

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
      //TODO: Create functions for study data
    loadAfterRankingB: function(req,res,user) {
      co(function* () {
        
        //connect to db
        let client = yield MongoClient.connect(url);
        const db = client.db('ratingsrankingsframes')
        let usersCol = db.collection('users')
        let responseCol = db.collection('responses')


        //TODO: Make this account for a and b? 
        //NOTE: Might be a bug here

        check =  yield responseCol.findOne({"user": user.id, "collection": String(user.activityID), "type": 'ranking', "study": user.study()})

        if (check === null){
          res.render('rankings2', {userID : user.id, id: user.activityID , type: "rankings", frames: user.frames(), question: user.question(), error: "ERROR: Please submit a complete ranking"})
          return;
        } else{
          res.render('ratings2', {userID: user.id, id: user.activityID, type: "ratings", picture: 0, question: user.question()});
        }

      })

    },

    loaAfterRatingA: function(req, res, user, picture){

      if(parseInt(picture) === user.frames() - 1){
        picture === 0

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db('ratingsrankingsframes')
          let usersCol = db.collection('users')
          let responseCol = db.collection('responses')

          //load next question
          questionOrder = user.getQuestionOrderA()
          question2load = questionOrder[user.activityID - 1]
          questionLength = question2load.length

          indexOrder = user.getIndexOrderA()
          questionIndex = indexOrder[0]
          currentBatch = questionIndex[0]
          user.saveCurrentQuestion("a",JSON.stringify(question2load), currentBatch, questionLength)
          
          //adjust to next activity

          res.render('rankings', {userID: user.id, id: user.activityID , frames: user.frames(), type: "rankings", question: user.question()})
      
        });

      } else {
        picture = parseInt(picture)+ 1
        res.render('ratings', {userID: user.id, id: user.activityID, type: "ratings", picture, question: user.question()})
      }
    },

    loaAfterRatingB: function(req, res, user, picture){

      //reset if ratings are complete
      if(parseInt(picture) === user.frames() - 1){
        picture === 0

        co(function* () {

          let client = yield MongoClient.connect(url);
          const db = client.db('ratingsrankingsframes')
          let usersCol = db.collection('users')
          let responseCol = db.collection('responses')

          //load next question
          questionOrder = user.getQuestionOrderB()
          question2load = questionOrder[user.activityID - 1]
          questionLength = question2load.length

          indexOrder = user.getIndexOrderB()
          questionIndex = indexOrder[0]
          currentBatch = questionIndex[0]
          user.saveCurrentQuestion("b", JSON.stringify(question2load), currentBatch, questionLength)
          
          //adjust to next activity

          res.render('rankings2', {userID: user.id, id: user.activityID , frames: user.frames(), type: "rankings", question: user.question()})
      
        });

      } else {
        picture = parseInt(picture)+ 1
        res.render('ratings2', {userID: user.id, id: user.activityID, type: "ratings", picture, question: user.question()})
      }
    }

}

module.exports = loadModule
