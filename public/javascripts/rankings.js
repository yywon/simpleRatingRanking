function renderQuestion(question, id, userID){

    var startTime = new Date().getTime();

    groupnum = question[0]
    candidates = question[1]

    console.log("groupnum    " + groupnum)
    console.log("candidates   " + candidates)

    //hook onto html element
    d3.select(".activity").html("")

    //Variables
    topNSize = 4
    topNPositions = Array.from(Array(topNSize).keys())
    var rankingOrder = []

    //Global Variables to be deleted

    d3.select(".activitypage .activity").append('div').attr("class", "label4ranking")
    d3.select(".activitypage .activity").append('div').attr("class", "rankingDiv")
    d3.select(".activitypage .activity").append('div').attr("class", "candidatePoolContents")
    //d3.select(".candidatePoolContainer").append('div').attr("class", "candidatePool")

    //let margin = { top: 2, right: 0, bottom: 0, left: 2 }
    let svg4ranking_width = "100%";
    let divWidth = d3.select('.rankingDiv').node().offsetWidth
    let rankingImageSize = divWidth / (topNSize +.5)
    let svg4ranking_height = rankingImageSize + 70;
    let gap4images = (divWidth - (rankingImageSize * topNSize)) / topNSize

    //Grouping for ranks

    let svg4ranking = d3.select(".rankingDiv").append("svg")
        .attr("width", svg4ranking_width)
        .attr("height", svg4ranking_height)
        //.attr("transform", "translate(" + margin.left + "," + margin.top + ")")

        let g4rankingborder = svg4ranking.append("g")
        .attr("class", "g4rankingborder")
        .selectAll("rect")
        .data(topNPositions)
        .enter().append("rect")
        .attr("x", function (d, i) {
            let xValue = (rankingImageSize + gap4images) * i
            return xValue
        })
        .attr("y", 0)
        .attr("width", rankingImageSize + 4)
        .attr("height", rankingImageSize + 4)
        .attr("style", function (d, i) {
            return "fill:transparent;stroke:black;stroke-width:1;stroke-opacity:1"
        })
    

    arr4label = ["Least Amount of Dots", "Most Amount of Dots"]
    labelDivWidth = (rankingImageSize + gap4images) + "px"

    let g4rankingImage = svg4ranking.append("g")
        .attr("class", "g4rankingImage")

    d3.select(".label4ranking").selectAll(".label1")
        .attr("class", "ranking")
        .style("width", labelDivWidth)
        .html("asdsa")
    
    
    //Grouping for candidates
    //let margin4pool = { top: 10, right: 0, bottom: 0, left: 0 }
    let svg4pool_width = "100%";
    let svg4pool_height = (rankingImageSize) + 50;
    let svg4pool = d3.select(".candidatePoolContents").append("svg")
        .attr("width", svg4pool_width)
        .attr("height", svg4pool_height)
        //.attr("transform", "translate(" + margin4pool.left + "," + margin4pool.top + ")")
    
        let g4candidatesBorder = svg4pool.append("g")
            .attr("class", "g4candidatesBorder")
            .selectAll("rect")
            .data(topNPositions)
            .enter().append("rect")
            .attr("x", function (d, i) {
                let xValue = (rankingImageSize + gap4images) * i
                return xValue
            })

        .attr("y", function (d, i) {
            let yValue = 0
            return yValue;
        })

        .attr("width", rankingImageSize + 4)
        .attr("height", rankingImageSize + 4)
        .attr("style", "fill:transparent;stroke:gray;stroke-width:1;stroke-opacity:1")

    let g4candidatesImage = svg4pool.append("g")
        .attr("class", "g4candidatesImage")
        .selectAll("image")
        .data(candidates)
        .enter().append("image")
        .attr('xlink:href', function (d, i) {
            let path4image = "/images/dots/" + groupnum + "/" + d + ".png"
            return path4image
        })
        .attr("x", function (d, i) {
            let xValue = 2 + (rankingImageSize +  gap4images) * i
            return xValue
        })
        .attr("y", function (d, i) {
            let yValue = 2
            return yValue 
        })

        .attr("width", rankingImageSize)
        .attr("height", rankingImageSize)
        .on("mouseover", function() {
            let hovereditem = this;
            let hoveredX = d3.select(hovereditem).attr("x")
            let hoveredY = d3.select(hovereditem).attr("y")
            let hoveredImage = d3.select(hovereditem).attr("href")
            //console.log("hovereditem ", hoveredImage)
            //add border to hovered heatmap
            svg4pool.append("rect")
                .attr("class", "hoveredborder4Pool")
                .attr("x", hoveredX - 2)
                .attr("y", hoveredY - 2)
                .attr("width", rankingImageSize + 4)
                .attr("height", rankingImageSize + 4)
                .attr("style", "fill:transparent;stroke:blue;stroke-width:2;stroke-opacity:1")
                .attr("pointer-events", "none")

            //try to put the image to the ranking list

            //find existing ranking
            existingRanking = rankingOrder.length

            //check to see if ranking list is full
            if (existingRanking > topNSize) {
                console.log("do nothing because the ranking list is full.")
            } else {
                g4rankingImage.append("image")
                    .attr("class", "tryRankingCandidate")
                    .attr('xlink:href', hoveredImage)
                    .attr("x", function () {
                        let position2put = (rankingImageSize + gap4images) * existingRanking + 2
                        return position2put
                    })
                    .attr("y", 2)
                    .attr("width", rankingImageSize)
                    .attr("height", rankingImageSize)
            }
        })
        .on("mouseout", function () {
                if (d3.select(".hoveredborder4Pool")) d3.select(".hoveredborder4Pool").remove();
                if (d3.select(".tryRankingCandidate")) d3.select(".tryRankingCandidate").remove();
        })
        .on("click", function () {
            console.log("click1")

            let clickeditem = this;
            let clickedX = d3.select(clickeditem).attr("x")
            let clickedY = d3.select(clickeditem).attr("y")
            let clickedImage = d3.select(clickeditem).attr("href")
            let splitName = clickedImage.split("/")
            let imageFullname = splitName[splitName.length - 1]
            let splitedFullname = imageFullname.split(".")
            let imageIndex4data = splitedFullname[0]
            
            //let existingRanking = d3.selectAll(".CANofRanking")._groups[0].length
            let existingRanking = rankingOrder.length

            console.log("existingRanking to add the image to rankinglist ", existingRanking)
            //add image to ranking array
            rankingOrder.push(imageIndex4data)
            //send data if full
            if (rankingOrder.length > 3){
                var endTime = new Date().getTime();
                var timeSpent = endTime- startTime;
                sendData(rankingOrder, id, userID, timeSpent)
            }

            svg4pool.append("rect")
                .attr("class", "clickedborder4Pool_" + imageIndex4data)
                .attr("x", clickedX - 2)
                .attr("y", clickedY - 2)
                .attr("width", rankingImageSize + 2)
                .attr("height", rankingImageSize + 2)
                .attr("style", "fill:gray;stroke:gray;stroke-width:2;stroke-opacity:1;opacity:0.5")

                .on("click", function () {

                    console.log("click2")
                    reset(rankingOrder);

                })

            /////put the clicked heatmap of the pool to ranking list
            g4rankingImage.append("image")
            .attr("class", "rankCandidate_" + imageIndex4data)
            .attr('xlink:href', clickedImage)
            .attr("x", function () {
                let existingRanking = rankingOrder.length - 1
                let position2put = (rankingImageSize + gap4images) * existingRanking + 2
                return position2put
            })
            .attr("y", 2)
            .attr("width", rankingImageSize)
            .attr("height", rankingImageSize)
    
        });

    //var timeSpentReport = TimeMe.getTimeOnCurrentPageInSeconds();

}
            
function sendData(rankingOrder, id, userID, time){
    console.log("sending data")
    console.log("Time: ", time)
    
    //url2go =  id + "/rankings"
    url2go = userID + "/" + id + "/sendRankings/"

    //add time to end of rankingOrder array
    group = rankingOrder
    group.push(time)

    //add ajax function
    new Promise((resolve, reject) => {
            $.ajax({
                dataType: "json",
                url: url2go,
                type: "POST",
                data: JSON.stringify(group),
                success: resolve
            });
        });
}

function reset(array){
    i = array.length;
    while(i > 0){

        image = array[i-1];
         //remove image
        if (d3.select(".clickedborder4Pool_"  + image)) d3.select(".clickedborder4Pool_"  + image).remove();
        if (d3.select(".rankCandidate_" + image)) d3.select(".rankCandidate_" + image).remove();
        array.pop()
        i--;

    }
}
