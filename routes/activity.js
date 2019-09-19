var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';
var userID = null
let assignQuestions = require('./assignQuestions')
let loadQuestion = require('./loadQuestion')

//store userID and load first activity
router.post('/', function(req,res,next){

  userID = req.body.userID ? req.body.userID : userID
  id = 1

  co(function* () {

    let client = yield MongoClient.connect(url);
    const db = client.db('ratingsrankingsbasic')
    let usersCol = db.collection('users') 

    let assignedQuestions = assignQuestions.assign();

    check = yield usersCol.findOne({"user" : userID})

    //check to see if user exists in database
    if(check === null){
      
      //insert new user if user does not exist
      var item = { 
         "user": userID,
         "key2pay": null,
         "group4Answers": assignedQuestions
      };

      yield usersCol.insertOne(item);

       //load next question
      question = loadQuestion.load(userID, id)

      //Console.log("QUESTION : " +   question)
      //json encode array
      question = JSON.stringify(question)

      res.render('rankings', {userID, id, type: "rankings", question})


    }
  });

 
});

//post a ranking
router.post(':s?/:t?/:d?/:userID/:id/sendRankings/', function(req,res,next){

  //collect variables
  userID = req.params.userID
  id = req.params.id;
  let group2save = Object.keys(req.body);
  group2save = JSON.parse(group2save)

  console.log(userID)
  console.log(id)

  //store into db
  co(function* () {

    let client = yield MongoClient.connect(url);
    const db = client.db('ratingsrankingsbasic')
    let responseCol = db.collection('responses')

    var item = {
      "id" : userID,
      "collection": id,
      "type": "ranking",
      "pos0": parseInt(group2save[0]),
      "pos1": parseInt(group2save[1]),
      "pos2": parseInt(group2save[2]),
      "pos3": parseInt(group2save[3])
    }

    var criteria = {
      "id": userID, 
      "collection": id, 
      "type": "ranking"
    }

    var newItem = {
        "pos0": parseInt(group2save[0]),
        "pos1": parseInt(group2save[1]),
        "pos2": parseInt(group2save[2]),
        "pos3": parseInt(group2save[3])
    }

    count = yield responseCol.find(criteria).count()
    console.log(count)

    if(count > 0){
      responseCol.update(criteria,{ $set: newItem })
      console.log('Ranking updated')
    } else {
      responseCol.insertOne(item, function(err, result) {
        console.log('Ranking inserted')
      });
    }

    client.close();
      
  });
});

router.post('/:id/rankings/', function(req, res, next){

  userID = req.body.userID ? req.body.userID : userID
  id = req.params.id;

  console.log("user", userID);

  question = loadQuestion.load(userID, id)

  //json encode array
  question = JSON.stringify(question)

  res.render('ratings', {userID, id, type: "ratings", picture: 0, question})

});

//Post a rating and load next page
router.post('/:id/ratings/:picture', function(req,res,next){

  //collect variables
  userID = req.body.userID ? req.body.userID : userID;
  rating = req.body.rating;
  id = req.params.id;
  picture = req.params.picture;

  //insert rating into db
  co(function* () {

    let client = yield MongoClient.connect(url);
    const db = client.db('ratingsrankingsbasic')
    let responseCol = db.collection('responses')

    var item = {
      "id" : userID,
      "collection": id,
      "type": "rating",
      "picture": picture,
      "estimate": rating
    }

    responseCol.insertOne(item, function(err, result) {
      console.log('Rating inserted')
      console.log("Inserted id:" + id)
      console.log("Inserted picture:" + picture)

      if(parseInt(id) === 7 && parseInt(picture) === 3){
        console.log('rendering survey')
        res.render('survey')
        return;
      }
    
      //adjust to next activity
      if(parseInt(picture) === 3){
        console.log("moving to next id")
        picture === 0
        id = parseInt(id) + 1
        type = "rankings"
        question = loadQuestion.load(userID, id)
        //json encode array
        question = JSON.stringify(question)
        res.render('rankings', {userID, id, type, question})
      } else {
        picture = parseInt(picture)+ 1
        question = loadQuestion.load(userID, id)
        //json encode array
        question = JSON.stringify(question)
        res.render('ratings', {userID, id, type: "ratings", picture, question})
      }

    });
  //go to survey if activity is finished

});

});

module.exports = router;

