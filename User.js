module.exports = class User {
    constructor(id) {
        this.id = id;
        this.activityID = 1;
        this.studyQuestion = 1;
        this.abOrder = ""
    }

    setStudyQuestion(num){
        this.studyQuestion = num
    }

    saveQuestionOrder(order){
        this.questionOrderA = order
    }

    saveIndexOrder(order){
        this.indexOrderA = order
    }

    saveCurrentQuestion(question, batch, length) {
        this.currentQuestion = question;
        this.currentBatch = batch
        this.currentFrames = length
    }
    
    question() {
        return this.currentQuestion;
    }

    batch() {
        return this.currentBatch
    }

    frames(){
        return this.currentFrames
    }

    getQuestionOrder() {
        return this.questionOrderA;
    }

    getIndexOrder() {
        return this.indexOrderA;
    }
   
};