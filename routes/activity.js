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
let assignQuestions = require('./assignQuestions')

//User Objects and current User array
const User = require('../User');
let users = [];

//Batch object
const Batch = require('../batch');

//populate batches
masterBatch = []
batch3 = []
batch4 = []
batch5 = []
batch6 = []

let i= 0
while(i < 9){
  batch3.push(new Batch(3))
  i++
}
masterBatch.push(batch3)

let j= 0
while(j < 12){
  batch4.push(new Batch(4))
  j++
}
masterBatch.push(batch4)

let k= 0
while(k < 15){
  batch5.push(new Batch(5))
  k++
}
masterBatch.push(batch5)

let l= 0
while(l < 18){
  batch6.push(new Batch(6))
  l++
}
masterBatch.push(batch6)

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
  //assign questions
  assignedQuestions = []
  assignedIndexes = []
  //iterate through batches
  for(i = 0; i < userOrder.length; i++){
      frame = userOrder[i]
      //Minus 3 to get frame index in master batch
      frameIndex = frame - 3
      frameLevel = masterBatch[frameIndex]
      console.log(frameLevel)

      findQuestions:
      //find first unassigned question in frameLevel
      for(batch = 0; batch < frameLevel.length; batch++){
          for(question = 0; question < frameLevel[batch].assignmentStatus.length; question++){
              if(frameLevel[batch].assignmentStatus[question] === 0){
                  //update assignment status 
                  frameLevel[batch].assignmentStatus[question] = 1
                  //get question from frameLevel and add to questions array
                  assignedQuestions.push(frameLevel[batch].questions[question])
                  assignedIndexes.push([batch,question])
                  break findQuestions;
              }
          }
      }

  }

  console.log("assigned Quesions: ", assignedQuestions)
  console.log("assigned Indexes: ", assignedIndexes)

  currentUser.saveQuestionOrder(assignedQuestions)
  currentUser.saveIndexOrder(assignedIndexes)

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

//load next rating page
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
  if(currentUser.activityID === 8){
    res.render('survey', {userID: currentUser.id})
    return
  } 
    //load new question
    loadQuestion.loadAfterRating(req, res, currentUser, picture);
  
});

module.exports = router;

