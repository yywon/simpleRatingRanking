module.exports = class User {

    centuries = [1400, 1500, 1600, 1700, 1800, 1900, 2000]

    array_1400s = 




    constructor(id) {
        this.id = id;
        this.activityID = 1;
    }

    saveCurrentQuestion(question) {
        this.currentQuestion = question;
    }

    question() {
        return this.currentQuestion;
    }
};