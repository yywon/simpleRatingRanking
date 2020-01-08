var express = require('express');
var router = express.Router();
var shuffle = require('shuffle-array');
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

const User = require('../User');
const Batch = require('../batch');


const assignModule = {

    assign: function(url){

        co(function* () {

            let client = yield MongoClient.connect(url);
            const db = client.db('ratingsrankingsframes')
            let batchesCol = db.collection('batches')

            let i= 0
            while(i < 9){

                b = new Batch(3)

                var item = {
                    "size": b.size,
                    "number": i,
                    "completness": b.completness,
                    "assignmentStatus": b.assignmentStatus,
                    "questions": b.questions
                }

                batchesCol.insertOne(item, function(err, result) {
                    //console.log('Ranking inserted')
                });

                i++
            }

            let j= 0
            while(j < 12){

                b = new Batch(4)

                var item = {
                    "size": b.size,
                    "number": j,
                    "completness": b.completness,
                    "assignmentStatus": b.assignmentStatus,
                    "questions": b.questions
                }

                batchesCol.insertOne(item, function(err, result) {
                    //console.log('Ranking inserted')
                });

                j++

            }


            let k= 0
            while(k < 15){

                b = new Batch(5)

                var item = {
                    "size": b.size,
                    "number": k,
                    "completness": b.completness,
                    "assignmentStatus": b.assignmentStatus,
                    "questions": b.questions
                }

                batchesCol.insertOne(item, function(err, result) {
                    //console.log('Ranking inserted')
                });

                k++
            }


            let l= 0
            while(l < 18){

                b = new Batch(6)

                var item = {
                    "size": b.size,
                    "number": l, 
                    "completness": b.completness,
                    "assignmentStatus": b.assignmentStatus,
                    "questions": b.questions
                }

                batchesCol.insertOne(item, function(err, result) {
                    //console.log('Ranking inserted')
                });

                l++
            }
        })
    }
}

module.exports = assignModule 