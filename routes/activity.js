var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');
var shuffle = require('shuffle-array');

var url = 'mongodb://localhost:27017/';
//var url = 'mongodb://10.218.105.218:27017/';

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
assignBatch.assign(url)

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
    userOrder = shuffle([2,3,5,6]);
    console.log("order: ", userOrder)

    //load first question
    loadQuestion.loadFirst(req, res, currentUser, userOrder)

});

//load new rating question
router.post('/:id/rankings/:userID', function(req, res, next){

  setTimeout(function (){
    //Fetch instance of current user
    let currentUser = getUserInstance(req.params.userID);
    loadQuestion.loadAfterRanking(req, res, currentUser);
  }, 250)

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

  //get rid of extra time variable in the group (so that group only constains ranking)
  group.pop()

  //store ranking
  storeQuestion.storeRanking(userID, id, group, time, frames, batch)

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

  console.log("in the funcccc")

  //collect variables from front end
  userID = req.params.userID
  id = req.params.id;
  picture = req.params.picture;
  let data = Object.keys(req.body);
  data = JSON.parse(data)

  let time = data[0]
  let rating = data[1]
  console.log('ratings', rating)

  let currentUser = getUserInstance(userID);
  let batch = currentUser.batch();
  let frames = currentUser.frames();
  
  var ratingInt = rating.map(function (x){
    return parseInt(x)
  })

  for(var i = 0; i < frames; i++){
    if(isNaN(ratingInt[i]) || ratingInt[i].trim() === ''){
      return;
    }
  }

  console.log('made it')
  storeQuestion.storeMultipleRatings(userID, id, rating, time, batch, frames)

});

router.post('/:id/ratingsB/B/:userID', function(req,res,next){

  setTimeout(function (){
    let currentUser = getUserInstance(req.params.userID);
    frames = currentUser.frames()

    loadQuestion.loadAfterRating(req, res, currentUser)

  }, 250)
})

module.exports = router;

