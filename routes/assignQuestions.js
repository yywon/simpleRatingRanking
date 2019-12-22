var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

const User = require('../User');

const assignModule = {

    assignOrder: function(){

        frames = [3,4,5,6]
        shuffle(frames)

        return frames
    }
}

module.exports = assignModule 