var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');
var shuffle = require('shuffle-array');

//var url = 'mongodb://localhost:27017/';
var url = 'mongodb://10.218.105.218:27017/';

var userID = null
let loadQuestion = require('./loadQuestion')
let storeQuestion = require('./storeQuestion')
let assignBatch = require('./assignBatch')

//User Objects and current User array
const User = require('../User');
let users = [];

//Batch object
const Batch = require('../batch');

//assign batches
//NOTE: comment this out if batches are already populated
//assignBatch.assign(url)

//function to get current issues of Users
let getUserInstance = uid => users.find(user => user.id === uid);

//store userID and load first activity
router.post('/', function(req,res,next){

    //prompt to enter username if null
    if (!req.body.userID) {
      res.render('index', {error: "ERROR: Please enter a username"});
      return;
    }

    //Fetch current user
    let currentUser = getUserInstance(req.body.userID);
    
    //add new user if not already exists based on id
    if (!currentUser) {
      users.push(new User(req.body.userID));
      currentUser = getUserInstance(req.body.userID);
    }
    
    //assign order of frames seen
    userOrder = shuffle([3,4,5,6]);
    console.log("order: ", userOrder)

    //load first question
    loadQuestion.loadFirst(req, res, currentUser, userOrder)

});

//load new rating question
router.post('/:id/rankings/:userID', function(req, res, next){

  //Fetch instance of current user
  let currentUser = getUserInstance(req.params.userID);

  //TODO: Make if else statement for study
  if(currentUser.study() === "a"){
    //load next question
    loadQuestion.loadAfterRankingA(req, res, currentUser);
  } else{
    loadQuestion.loadAfterRankingB(req, res, currentUser);
  }

});


//post a ranking
router.post(':s?/:t?/:d?/:f?/:userID/:id/sendRankings/', function(req,res,next){

  //collect variables
  userID = req.params.userID;
  id = req.params.id;
  let group = Object.keys(req.body);
  group = JSON.parse(group)
  time = group[group.length - 1]

  console.log(group)

  let currentUser = getUserInstance(userID);
  let batch = currentUser.batch();
  let frames = currentUser.frames();
  let study = currentUser.study();

  //get rid of extra time variable in the group (so that group only constains ranking)
  group.pop()

  //store ranking
  storeQuestion.storeRanking(userID, id, group, time, frames, batch, study)

});



//send survey questions
router.post('/:s?/:t?/:d?/:f?/:userID/sendSurvey', function(req,res,next){

  //collect variables from front end
  userID = req.params.userID;
  key = req.body.key;
  userDemographic = req.body.userDemographic;
  userDemographic = JSON.parse(userDemographic);

  //storesurvey results
  storeQuestion.storeSurvey(userID, userDemographic, key)

  //give a response to load next page
  res.send("{}");

})

//send ratings A
router.post(':s?/:t?/:d?/:f?/:userID/:id/:picture/sendRatings/', function(req,res,next){

  //collect variables from front end
  userID = req.params.userID
  id = req.params.id;
  picture = req.params.picture;
  let data = Object.keys(req.body);
  data = JSON.parse(data)

  let time = data[0]
  let rating = data[1]

  //return if rating is not valid
  if(isNaN(rating) || rating === ''){
    return;
  }

  let currentUser = getUserInstance(userID);
  let batch = currentUser.batch();
  let frames = currentUser.frames();

  //store if rating is valid input
  storeQuestion.storeRating(userID, id, picture, rating, time, batch, frames)

});

//load next rating page (will only be called in study a)
router.post('/:id/ratings/:picture/:userID', function(req,res,next){

  //collect variables
  rating = req.body.rating;
  time = req.body.time;
  id = req.params.id;
  picture = req.params.picture;

  //Fetch current user instance
  let currentUser = getUserInstance(req.params.userID);
  console.log(currentUser)

  //render next page if input is valid
  if(isNaN(rating) || rating === ''){
    res.render('ratings', { userID: currentUser.id , id: currentUser.activityID , type: "ratings", picture, question: currentUser.question(), error: "ERROR: Please submit a valid estimate"})
    return;
  }

  //increment activity ID if user makes it to the final picture
  if(parseInt(picture) === currentUser.frames() - 1){
    currentUser.activityID += 1
  }

  //load survey if activity is complete
  if(currentUser.activityID === 5 || currentUser.activityID === 9){
    loadQuestion.loadNextStudy(req,res,currentUser)
    return;
  } 
    
  loadQuestion.loadAfterRatingA(req, res, currentUser, picture);

});

//TODO: ratings B routing 

//send ratings B
router.post(':s?/:t?/:d?/:f?/:userID/:id/B/sendRatings/', function(req,res,next){


  storeQuestion.storeMultipleRatings()
  
});


router.post('/:id/ratings/B/:userID/B', function(req,res,next){

  //get ratings from req.body

  let currentUser = getUserInstance(req.params.userID);
  currentUser.activityID += 1

  if(currentUser.activityID === 5 || currentUser.activityID === 9){
    loadQuestion.loadNextStudy(req,res,currentUser)
    return;
  } 

  loadQuestion.loadAfterRatingB(req, res, currentUser)

})

module.exports = router;

