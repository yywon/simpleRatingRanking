var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

const assignModule = {

    assign: function() {
        
        questions = []
        for(i = 0; i < 8; i++){
            q = []
            for(j = 0; j < 4; j++){
                questionNumber = 50 + Math.floor(Math.random() * 40);
                q[j] = questionNumber
            }
            questions[i] = q
        }

    return questions
    }
}

module.exports = assignModule 