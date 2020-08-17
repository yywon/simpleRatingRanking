var express = require('express');
var router = express.Router();
var shuffle = require('shuffle-array');
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

const User = require('../User');
const Batch = require('../batch');

var dbase = 'ratingsrankingsB3'

const assignModule = {

    assign: function(url){

        co(function* () {

            let client = yield MongoClient.connect(url);
            const db = client.db(dbase)
            let batchesCol = db.collection('batches')

            let i= 0
            while(i < 9){

                b1 = new Batch(2)

                var itemA = {
                    "size": b1.size,
                    "number": i,
                    "completness": b1.completness,
                    "assignmentStatus": b1.assignmentStatus,
                    "questions": b1.questions
                }

                batchesCol.insertOne(itemA, function(err, result) {
                    //console.log('Ranking inserted')
                });

                i++
            }

            let j= 0
            while(j < 12){

                b1 = new Batch(3)

                var itemA = {
                    "size": b1.size,
                    "number": j,
                    "completness": b1.completness,
                    "assignmentStatus": b1.assignmentStatus,
                    "questions": b1.questions
                }


                batchesCol.insertOne(itemA, function(err, result) {
                    //console.log('Ranking inserted')
                });

                j++

            }


            let k= 0
            while(k < 15){

                b1 = new Batch(5)

                var itemA = {
                    "size": b1.size,
                    "number": k,
                    "completness": b1.completness,
                    "assignmentStatus": b1.assignmentStatus,
                    "questions": b1.questions
                }

                batchesCol.insertOne(itemA, function(err, result) {
                    //console.log('Ranking inserted')
                });

                k++
            }


            let l= 0
            while(l < 18){

                b1 = new Batch(6)

                var itemA = {
                    "size": b1.size,
                    "number": l,
                    "completness": b1.completness,
                    "assignmentStatus": b1.assignmentStatus,
                    "questions": b1.questions
                }

                batchesCol.insertOne(itemA, function(err, result) {
                    //console.log('Ranking inserted')
                });

                l++
            }
        })
    }
}

module.exports = assignModule 
