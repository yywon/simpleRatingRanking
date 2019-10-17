module.exports = class User {
    constructor(id) {
        this.id = id;
        this.activityID = 1;
    }

    saveCurrentQuestion(question) {
        this.currentQuestion = question;
    }

    get question() {
        return this.currentQuestion;
    }
};