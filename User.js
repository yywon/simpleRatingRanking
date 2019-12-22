module.exports = class User {
    constructor(id) {
        this.id = id;
        this.activityID = 1;
    }

    saveQuestionOrder(order){
        this.questionOrder = order
    }

    saveIndexOrder(order){
        this.indexOrder = order
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
        return this.questionOrder;
    }

    getIndexOrder() {
        return this.indexOrder;
    }

    
};