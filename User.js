module.exports = class User {
    constructor(uid) {
        this.uid = uid;
        this.activityID = 1;
    }

    saveCurrentQuestion(question) {
        this.currentQuestion = question;
    }

    get question() {
        return this.currentQuestion;
    }
};