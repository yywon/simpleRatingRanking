module.exports = class batch {
    
    constructor(size){
        this.size = size
        this.completeness = 60/size
        this.assignmentStatus = [0]*this.completeness
        this.questions = this.assignQuestions
    }

    assignQuestions(){

        let questions
        let allNumbers = [];
        for(i = 0; i < 60; i++){
            allNumbers[i] = 50 + i
        }

        shuffle(allNumbers)

        chunk = this.size
        i = 0;
        j = allNumbers.length; 

        while(i < j){
            temparray = allNumbers.slice(i,i+chunk);
            questions.push(temparray)
            i += chunk
        }

        return questions
    }
}
