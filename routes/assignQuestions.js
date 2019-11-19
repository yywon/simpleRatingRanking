var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

const assignModule = {

    assign: function() {
        
        //populate array
        allNumbers = []
        for(i = 0; i < 32; i++){
            allNumbers[i] = 50 + i
        }

        shuffle(allNumbers)

        questions = []
        chunk = 4
        length = allNumbers.length
        for(i = 0; i<length; i+=chunk){
            q = []
            q = allNumbers.slice(i, i+chunk)
            questions.push(q)
        }

    return questions
    }
}

module.exports = assignModule 