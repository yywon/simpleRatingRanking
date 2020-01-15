function renderQuestion(question, id, userID, frames){

    console.log(question)

    var startTime = new Date().getTime();
    
    topNSize = parseInt(frames)
    topNPositions = Array.from(Array(topNSize).keys()) //array from 1 to 4

    var rankingImageSize
    var gap4images
    var space

    d3.select(".activity").html("")

    d3.select(".activitypage .activity").append('div').attr("class", "images")
    d3.select(".activitypage .activity").append('div').attr("class", "estimations")

    let divWidth = d3.select('.activity').node().offsetWidth
    //specify size based on image gaps 
    if(topNSize === 3){
        rankingImageSize = divWidth/5;
        space = rankingImageSize;
        gap4images = (divWidth - (space + (rankingImageSize * topNSize)))/ topNSize;
    } else if(topNSize === 3){
        rankingImageSize = divWidth/5;
        space = rankingImageSize * 3/5;
        gap4images = (divWidth - (space + (rankingImageSize * topNSize)))/ topNSize;
    } else{
        rankingImageSize = divWidth / (topNSize + 1)
        space = rankingImageSize/8;
        gap4images = (divWidth - (space + (rankingImageSize * topNSize))) / topNSize;
    }

    let svgimagewidth = "100%"
    let imageSize = divWidth / 3
    let svg4rating_height = rankingImageSize + rankingImageSize/4;

    //grouping for image
    let svg4image = d3.select('.images').append('svg')
        .attr("width", svgimagewidth)
        .attr("height", svg4rating_height)

        let g4rankingborder = svg4image.append("g")
        .attr("class", "g4rankingborder")
        .selectAll("rect")
        .data(topNPositions)
        .enter().append("rect")
        .attr("x", function (d, i) {
            let xValue = space + ((rankingImageSize + gap4images) * i)
            return xValue
        })
        .attr("y", rankingImageSize/8)
        .attr("width", rankingImageSize + 4)
        .attr("height", rankingImageSize + 4)
        .attr("style", function (d, i) {
            return "fill:transparent;stroke:black;stroke-width:1;stroke-opacity:1"
        })

        let g4candidatesImage = svg4image.append("g")
        .attr("class", "g4candidatesImage")
        .selectAll("image")
        .data(question)
        .enter().append("image")
        .attr('xlink:href', function (d, i) {
            let path4image = "/images/dots/" + d + ".png"
            return path4image
        })

        .attr("x", function (d, i) {
            let xValue = space + 2 + (rankingImageSize +  gap4images) * i
            return xValue
        })

        .attr("y", function (d, i) {
            let yValue = rankingImageSize/8 + 2
            return yValue 
        })

        .attr("width", rankingImageSize)
        .attr("height", rankingImageSize)

    d3.select(".btn.btn-success.nextBtn").on("click", function () {
        //console.log("Button Clicked");
        //console.log(userID)
        //console.log(id)
        var endTime = new Date().getTime();
        var timeSpent = endTime - startTime;
        rating = document.getElementById("rating").value
        sendData(id, userID, picture, timeSpent, rating)
    })
}

function sendData(id, userID, picture, time, rating){
    //console.log("sending data")
    
    //url2go =  id + "/rankings"
    url2go = userID + "/" + id + "/" + picture + "/sendRatings/"

    data2send = [time, rating]
    
    /*
    //add ajax function
    new Promise((resolve, reject) => {
            $.ajax({
                dataType: "json",
                url: url2go,
                type: "POST",
                data: JSON.stringify(data2send),
                success: resolve
            });
        });

    */
}