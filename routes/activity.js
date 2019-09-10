var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';
var userID = null

//store userID and load first activity
router.post('/', function(req,res,next){

  userID = req.body.userID ? req.body.userID : userID

  co(function* () {

    let client = yield MongoClient.connect(url);
    const db = client.db('ratingsrankingsbasic')
    let usersCol = db.collection('users')

    check = yield usersCol.findOne({"user" : userID})

    //check to see if user exists in database
    if(check === null){
      
      //insert new user if user does not exist
      var item = { "user": userID };

      usersCol.insertOne(item, function(err, result) {
        console.log('Username inserted', result);
      });
    }
  });

  res.render('rankings', {userID, id: 1 , type: "rankings"})

});


//post a ranking and load next rating page
router.post('/:id/rankings/', function(req,res,next){

  //collect variables
  userID = req.body.userID ? req.body.userID : userID;
  rankingOrder = 
  id = req.params.id;
  type = req.params.type;

  //store into db
  co(function* () {

    let client = yield MongoClient.connect(url);
    const db = client.db('ratingsrankingsbasic')
    let responseCol = db.collection('responses')

    var item = {
      "id" : userID,
      "collection": id,
      "type": "ranking",
      "pos0": rankingOrder[0],
      "pos1": rankingOrder[1],
      "pos2": rankingOrder[2],
      "pos3": rankingOrder[3]
    }

    responseCol.insertOne(item, function(err, result) {
      console.log('Ranking inserted')
    });
      
  });

  res.render('ratings', {userID, id, type: "ratings", picture: 0})

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
    });

  //go to survey if activity is finished
  if(parseInt(id) === 4 && parseInt(picture) === 3){
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
    res.render('rankings', {userID, id, type})
  } else {
    picture = parseInt(picture)+ 1
    res.render('ratings', {userID, id, type: "ratings", picture})
  }


  });

});

module.exports = router;

