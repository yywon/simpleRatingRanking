var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';
var userID = null
let loadQuestion = require('./loadQuestion')
let storeQuestion = require('./storeQuestion')

//User Objects and current User array
const User = require('../User');
let users = [];

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

  //load first question
  loadQuestion.loadFirst(req, res, currentUser)

});

//load new rating question
router.post('/:id/rankings/:userID', function(req, res, next){

  //Fetch instance of current user
  let currentUser = getUserInstance(req.params.userID);

  //load next question
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

  //get rid of extra time variable in the group (so that group only constains ranking)
  group.pop()

  //store ranking
  storeQuestion.storeRanking(userID, id, group, time)

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

//send ratings
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

  //store if rating is valid input
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

  //Fetch current user instance
  let currentUser = getUserInstance(req.params.userID);
  console.log(currentUser)

  //render next page if input is valid
  if(isNaN(rating) || rating === ''){
    res.render('ratings', { userID: currentUser.id , id: currentUser.activityID , type: "ratings", picture, question: currentUser.question(), noiselevel, error: "ERROR: Please submit a valid estimate"})
    return;
  }

  //increment activity ID if user makes it to 3rd picture
  if(parseInt(picture) === 3){
    currentUser.activityID += 1
  }

  //load survey if activity is complete
  if(currentUser.activityID === 9 && parseInt(picture) === 3){
    res.render('survey', {userID: currentUser.id })
    return
  } 
    //load new question
    loadQuestion.loadAfterRating(req, res, currentUser, picture);
  
});

module.exports = router;

