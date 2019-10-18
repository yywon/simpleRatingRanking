var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';
var userID = null
let loadQuestion = require('./loadQuestion')
let storeQuestion = require('./storeQuestion')

const User = require('../User');
let users = [];

let getUserInstance = uid => users.find(user => user.id === uid);

//store userID and load first activity
router.post('/', function(req,res,next){

  //NOTE: this is where the override is happening,
  //my suggestion would be to create a new instance for each user.
  //I think I suggested this approach initially. Sorry.

  if (!req.body.userID) {
    res.render('index', {error: "ERROR: Please enter a username"});
    return;
  }

  console.log(req.body)

  let currentUser = getUserInstance(req.body.userID);

  //NOTE: add new user if not already exists based on id
  if (!currentUser) {
    users.push(new User(req.body.userID));
    currentUser = getUserInstance(req.body.userID);
  }

  console.log("current User: ", currentUser)

  // loadQuestion.loadFirst(req, res, userID, id)
  //NOTE: pass in user's instance instead
  loadQuestion.loadFirst(req, res, currentUser)

});

//load new rating question
router.post('/:id/rankings/:userID', function(req, res, next){

  let currentUser = getUserInstance(req.params.userID);
  console.log(currentUser)

  loadQuestion.loadAfterRanking(req, res, currentUser);

});


//post a ranking
router.post(':s?/:t?/:d?/:f?/:userID/:id/sendRankings/', function(req,res,next){

  //collect variables
  userID = req.params.userID;
  id = req.params.id;
  let group = Object.keys(req.body);
  group = JSON.parse(group)
  time = group[4]

  //get rid of extra time variable in the group
  group.pop()
  storeQuestion.storeRanking(userID, id, group, time)

});



//send survey questions
router.post('/:s?/:t?/:d?/:f?/:userID/sendSurvey', function(req,res,next){

  userID = req.params.userID;
  key = req.body.key;
  userDemographic = req.body.userDemographic;
  userDemographic = JSON.parse(userDemographic);

  //console.log(userDemographic);

  storeQuestion.storeSurvey(userID, userDemographic, key)

  res.send("{}");

})

//send ratings
router.post(':s?/:t?/:d?/:f?/:userID/:id/:picture/sendRatings/', function(req,res,next){

  userID = req.params.userID
  id = req.params.id;
  picture = req.params.picture;

  let data = Object.keys(req.body);
  data = JSON.parse(data)

  //console.log(data)

  let time = data[0]
  let rating = data[1]

  //console.log("Time: ", time);
  //console.log("rating, ", rating)
  //console.log("user", userID);
  //console.log("id", id);

  if(isNaN(rating) || rating === ''){
    return;
  }

  storeQuestion.storeRating(userID, id, picture, rating, time)

});

//load next rating page
router.post('/:id/ratings/:picture/:userID', function(req,res,next){

  //collect variables
  //userID = req.body.userID ? req.body.userID : userID;
  rating = req.body.rating;
  time = req.body.time;
  id = req.params.id;
  picture = req.params.picture;

  let currentUser = getUserInstance(req.params.userID);

  if(isNaN(rating) || rating === ''){
    //TODO: question references global var, use current user's current question instead
    //currentUser.question
    res.render('ratings', { userID: currentUser.id , id: currentUser.activityID , type: "ratings", picture, question: currentUser.question, noiselevel, error: "ERROR: Please submit a valid estimate"})
    return;
  }

  if(parseInt(picture) === 3){
    currentUser.activityID += 1
  }

  if(parseInt(id) === 9 && parseInt(picture) === 3){
    res.render('survey', {userID: currentUser.id })
    return;
  }

  //load new question
  loadQuestion.loadAfterRating(req, res, currentUser, picture);

});

module.exports = router;

