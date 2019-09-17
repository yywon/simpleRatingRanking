var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

var url = 'mongodb://10.218.105.218:27017/';

//post a ranking
router.post('/:userID/:id/sendRankings/', function(req,res,next){

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