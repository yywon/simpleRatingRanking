var shuffle = require('shuffle-array');


module.exports = class batch {
    
    constructor(size){
        this.size = size
        this.completeness = 60/size
        this.assignmentStatus = []
        for(let i = 0; i < this.completeness; i++){
            this.assignmentStatus.push(0)
        }
        this.questions = this.assignQuestions()
    }

    assignQuestions(){

        let questions = [];
        let allNumbers = [];
        for(let i = 0; i < 60; i++){
            allNumbers[i] = 50 + i
        }

        shuffle(allNumbers)

        var chunk = this.size
        var i = 0;
        var j = allNumbers.length;
        var temparray = [] 

        while(i < j){
            temparray = allNumbers.slice(i,i+chunk);
            questions.push(temparray)
            i += chunk
        }

        return questions
    }

}
