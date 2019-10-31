module.exports = class User {
    constructor(id) {
        this.id = id;
        this.frames = frames;
        this.activityID = 1;
    }

    saveCurrentQuestion(question) {
        this.currentQuestion = question;
    }

    question() {
        return this.currentQuestion;
    }
};