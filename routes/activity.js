var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';
var userID = null
let loadQuestion = require('./loadQuestion')
let storeQuestion = require('./storeQuestion')

//store userID and load first activity
router.post('/', function(req,res,next){

  userID = req.body.userID ? req.body.userID : userID
  id = 1

  loadQuestion.loadFirst(req, res, userID, id)

});


//post a ranking
router.post(':s?/:t?/:d?/:userID/:id/sendRankings/', function(req,res,next){

  //collect variables
  userID = req.params.userID
  id = req.params.id;
  let group = Object.keys(req.body);
  group = JSON.parse(group)
  time = group[4]

  group.pop()
  storeQuestion.storeRanking(userID, id, group, time)
});


//load new rating question
router.post('/:id/rankings/', function(req, res, next){

  userID = req.body.userID ? req.body.userID : userID;
  id = req.params.id;

  loadQuestion.loadAfterRanking(req, res, userID, id);

});


router.post(':s?/:t?/:d?/:userID/:id/:picture/sendRatings/', function(req,res,next){
  console.log("YUPPPPPPPPPPPPP")

  userID = req.params.userID
  id = req.params.id;
  picture = req.params.picture;

  let data = Object.keys(req.body);
  data = JSON.parse(data)

  console.log(data)

  let time = data[0]
  let rating = data[1]

  console.log("Time: ", time);
  console.log("rating, ", rating)
  console.log("user", userID);
  console.log("id", id);

  storeQuestion.storeRating(userID, id, picture, rating, time)

});

//Post a rating and load next page
router.post('/:id/ratings/:picture', function(req,res,next){

  //collect variables
  userID = req.body.userID ? req.body.userID : userID;
  rating = req.body.rating;
  time = req.body.time;
  id = req.params.id;
  picture = req.params.picture;

  if(isNaN(rating) || rating === ''){
    res.render('ratings', {userID, id, type: "ratings", picture, question, noiselevel, error: "ERROR: Please submit a valid estimate"})
    return;
  }

  //storeQuestion.storeRating(userID, id, picture, rating)

  if(parseInt(picture) === 3){
    console.log("moving to next id")
    id = parseInt(id) + 1
  }

  if(parseInt(id) === 9 && parseInt(picture) === 3){
    console.log('rendering survey')
    res.render('survey')
    return;
  }

  //load new question
  loadQuestion.loadAfterRating(req, res, userID, id, picture);

});

module.exports = router;

