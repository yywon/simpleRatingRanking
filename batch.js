module.exports = class batch {
    constructor(size){
        this.size = 0
        this.complete = false
        this.users = []
    }

    addUser(User){
        this.users.push(User)
    }

}
