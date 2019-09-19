var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

const assignModule = {

    assign: function() {
        
        questions = []
        for(i = 0; i < 8; i++){
            questionNumber = Math.floor(Math.random() * 24);
            questions[i] = questionNumber
        }

        return questions
    }
}

module.exports = assignModule