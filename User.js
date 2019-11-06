module.exports = class User {
    constructor(id) {
        this.id = id;
        this.activityID = 1;
        this.frameNumber = 0;
    }

    saveCurrentQuestion(question) {
        this.currentQuestion = question;
    }

    saveFrames(frames){
        this.frameNumber = frames;
        this.total = 30/frames;
    }

    question() {
        return this.currentQuestion;
    }

    getFrames() {
        return this.frameNumber;
    }

    getTotal() {
        return this.total;
    }

    
};