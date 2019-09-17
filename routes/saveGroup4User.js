const MongoClient = require('mongodb').MongoClient,
    assert = require('assert');
const express = require('express');
const router = express.Router();
const co = require('co');

const url = 'mongodb://rwkemmer@10.218.105.218:22/ratingsrankingsbasic';

const saveGroup = {
    saveGroup = {},
    saveRes: function (req, res, next) {
        
        console.log("we did it")

        userID = req.body.userID
        id = req.params.id;
        let group2save = req.body.group2save

        co(function* () {

            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsbasic')
            let responseCol = db.collection('responses')
    
            var item = {
            "id" : userID,
            "collection": id,
            "type": "ranking",
            "pos0": group2save[0],
            "pos1": group2save[1],
            "pos2": group2save[2],
            "pos3": group2save[3]
            }
    
            responseCol.insertOne(item, function(err, result) {
                console.log('Ranking inserted')
            });

        db.close();
          
      });
    }
}

module.exports = saveGroup