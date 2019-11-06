var express = require('express');
var router = express.Router();
var MongoClient = require('mongodb').MongoClient;
var assert = require('assert');
const co = require('co');

const User = require('../User');

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
    },

    assignFrames: function(users){

        console.log(users)

        counts = [0,0,0,0]
        framesList = [2,3,5,6]

        for(i = 0; i < users.length; i++){
            currentUser = users[i]
            test = currentUser.getFrames();
            console.log("test: ", test)
            for(j = 0; j < framesList.length; j++){
                if(framesList[j] == test){
                counts[j] = counts[j] + 1
                }
            }
        }

        min = Math.min(...counts)
        index = counts.indexOf(min)
        userFrame = framesList[index]

        console.log(counts)

        return userFrame
    }


}

module.exports = assignModule 