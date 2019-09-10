const MongoClient = require('mongodb').MongoClient,
    assert = require('assert');
const express = require('express');
const router = express.Router();
const co = require('co');

const url = 'mongodb://rwkemmer@10.218.105.218:22/ratingsrankingbasic';
const users = "users"
const reference = "questionpool"
const studySize = 4

const storeId = {
    storeId: {},
    newUser: function (req, res, next) {
        let userid = req.body.userid
        // console.log("userid ", userid)
        co(function* () {
            let db = yield MongoClient.connect(url);
            let usersCol = db.collection(users)
            let referenceCol = db.collection(reference);

            //check user , insert or updata
            ////check
            let userExist = yield usersRankingCol.find({ "userId": userid }).toArray()
            if (userExist.length != 0) {
                let finishStudy = userExist[0].key2pay
                /************user exist and finish************* */
                if (finishStudy != "none") {
                    let key2show = yield usersRankingCol.find({ "userId": userid }).toArray()
                    key2show = key2show[0].key2pay
                    db.close();
                    res.send(JSON.stringify(key2show));
                } else {
                    /*********user exist and don't finish 1> tasks 2> demographic questions************** */
                    let groupsObj = userExist[0].groups2study
                    let allrefers = Object.keys(groupsObj)
                    let leftTasks = []
                    allrefers.forEach(d => {
                        if (groupsObj[d].visited == false) {
                            leftTasks.push(d)
                        }
                    })
                    // console.log("leftTasks ", leftTasks)
                   
                    if (leftTasks.length != 0) {
                    //////////don't finish 1> tasks
                        let ifLeft8 = leftTasks.includes("8")
                        if (ifLeft8) {
                            let index48 = leftTasks.indexOf("8");
                            if (index48 > -1) {
                                leftTasks.splice(index48, 1);
                            }
                        }

                        let tasks2send = yield reference100testingCol.find({
                            "referenceName": {
                                $in: leftTasks
                            }
                        }).toArray()

                        tasks2send = tasks2send.map(d => {
                            let obj = {}
                            let key = d.referenceName
                            let value = d.group4label
                            obj[key] = value
                            return obj
                        })
                        // console.log("tasks2send after ", tasks2send.length)
                        if (ifLeft8) {
                            tasks2send.unshift(standardGroup)
                        }
                        db.close();
                        res.send(JSON.stringify(tasks2send));
                    } else {
                    //////////don't finish 2> demographic questions
                        db.close();
                        res.send(JSON.stringify("Please finish the demographic questions"));
                    }

                }
            } else {
                /************************user doesn't exist ****************************** */
                function recordItem2updateVisitTime(refer2study, minVisitedTime, items2plus1) {
                    let querieditem = refer2study.map(d => {
                        return d.referenceName
                    })
                    let currentVisitedTime = minVisitedTime + 1
                    items2plus1.push([querieditem, currentVisitedTime])
                }
                let items2plus1 = []
                let arr2send = []
                let minVisitedTime = yield reference100testingCol.find({}, { "visitedTimes": 1 }).sort({ "visitedTimes": 1 }).limit(1).toArray()
                minVisitedTime = minVisitedTime[0].visitedTimes
                // console.log("minVisitedTime ", minVisitedTime)

                let refer2study = yield reference100testingCol.find({ "visitedTimes": minVisitedTime }).limit(studySize).toArray()
                recordItem2updateVisitTime(refer2study, minVisitedTime, items2plus1)
                arr2send = arr2send.concat(refer2study)

                while (studySize != arr2send.length) {
                    minVisitedTime++
                    let moreRefers = studySize - arr2send.length
                    let refers = yield reference100testingCol.find({ "visitedTimes": minVisitedTime }).limit(moreRefers).toArray()
                    recordItem2updateVisitTime(refers, minVisitedTime, items2plus1)
                    arr2send = arr2send.concat(refers)
                }


                ///////update visited Times for refers
                for (let index = 0; index < items2plus1.length; index++) {
                    const element = items2plus1[index];
                    let arr2update = element[0]
                    let updateValue = element[1]
                    let updateVisitedTimes = yield reference100testingCol.updateMany(
                        { "referenceName": { $in: arr2update } },
                        { $set: { "visitedTimes": updateValue } }
                    )
                }

                // console.log("arr2send ", arr2send)
                arr2send = arr2send.map(d => {
                    let obj = {}
                    let key = d.referenceName
                    let value = d.group4label
                    obj[key] = value
                    return obj
                })
                // console.log("arr2send !!! ", arr2send)
                arr2send.unshift(standardGroup) //add a standard question for each user

                ///////store user infor to usersranking collection
                let allrefers = arr2send.map(d => {
                    let result = Object.keys(d)
                    return result[0]
                })
                let groupsObj = {}
                allrefers.forEach(d => {
                    groupsObj[d] = { "ranked": [], "visited": false }
                })
                let obj2insert4user = { "userId": userid }
                obj2insert4user["key2pay"] = "none"
                obj2insert4user["groups2study"] = groupsObj
                let result = yield usersRankingCol.insert(obj2insert4user)
                db.close();

                res.send(JSON.stringify(arr2send));
            }



        })

    }
}

module.exports = storeId