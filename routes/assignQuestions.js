var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

const assignModule = {

    assign: function(frames) {
        
        //populate array
        allNumbers = []
        for(i = 0; i < 30; i++){
            allNumbers[i] = 50 + i
        }

        for(let i = allNumbers.length - 1; i > 0; i--){
            const j = Math.floor(Math.random() * i)
            const temp = allNumbers[i]
            allNumbers[i] = allNumbers[j]
            allNumbers[j] = temp
        }

        questions = []
        chunk = frames
        length = 30
        console.log(length)
        for(i = 0; i<length; i+=chunk){
            q = []
            q = allNumbers.slice(i, i+chunk)
            questions.push(q)
        }

    return questions
    }
}

module.exports = assignModule 