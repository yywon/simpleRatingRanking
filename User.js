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

    saveABOrder(order){
        this.abOrder = order
    }

    saveQuestionOrderA(order){
        this.questionOrderA = order
    }

    saveIndexOrderA(order){
        this.indexOrderA = order
    }

    saveQuestionOrderB(order){
        this.questionOrderB = order
    }

    saveIndexOrderB(order){
        this.indexOrderB = order
    }

    saveCurrentQuestion(study, question, batch, length) {
        this.currentStudy = study
        this.currentQuestion = question;
        this.currentBatch = batch
        this.currentFrames = length
    }

    study(){
        return this.currentStudy
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

    getQuestionOrderA() {
        return this.questionOrderA;
    }

    getIndexOrderA() {
        return this.indexOrderA;
    }

    getQuestionOrderB() {
        return this.questionOrderB;
    }

    getIndexOrderB() {
        return this.indexOrderB;
    }

    getABOrder() {
        return this.abOrder
    }

    
};