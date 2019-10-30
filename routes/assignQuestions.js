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

        for(let i = allNumbers.length - 1; i > 0; i--){
            const j = Math.floor(Math.random() * i)
            const temp = allNumbers[i]
            allNumbers[i] = allNumbers[j]
            allNumbers[j] = temp
        }

        questions = []
        chunk = 4
        length = allNumbers.length
        console.log(length)
        for(i = 0; i<length/2; i+=chunk){
            q = []
            q = allNumbers.slice(i, i+chunk)
            questions.push(q)
        }

    return questions
    }
}

module.exports = assignModule 