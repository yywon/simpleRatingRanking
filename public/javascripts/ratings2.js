//import { svg } from "d3";

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
    if(topNSize === 2){
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
        .attr("id", "svg4images")
        .attr("width", svgimagewidth)
        .attr("height", svg4rating_height)

        let g4rankingborder = svg4image.append("g")
        .attr("class", "g4rankingborder")
        .selectAll("rect")
        .data(topNPositions)
        .enter().append("rect")
        .attr("id", function(d,i){
            return i
        })
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
        .enter().append("xhtml:div")

        let g4candidatesImage = svg4image.append("g")
        .attr("class", "g4candidatesImage")
        .selectAll("image")
        .data(question)
        .enter().append("image")
        .attr('xlink:href', function (d, i) {
            let path4image = "/images/dots/dots" + topNSize + "/" + d + ".png"
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
    

    let est = d3.select(".estimations").append('svg')
        .attr("id", "estimations")
        .attr("width", svgimagewidth)
        .attr("height", 150)

    for(var i = 0; i < frames; i++){
        var str = i.toString();
        est.append('foreignObject')
            .attr("id","element"+str)
            .attr("x", (2 + space - 10 + (rankingImageSize +  gap4images) * i)) //Note took out "space + 2"
            .attr("y", 50)
            .attr("width", 200)
            .attr("height", 40)
        }


    for(var i = 0; i < frames; i++){

        var box = document.createElement("input")
        var str = i.toString();
        
        box.setAttribute("type", "text")
        box.setAttribute("class", "form-control")
        box.setAttribute("style", "margin-bottom: 30px; width: 150px; margin-left: auto; margin-right: auto;")
        box.setAttribute("autocomplete", "off")
        box.setAttribute("placeholder", "Enter Estimate")
        box.setAttribute("aria-label", "Enter Estimate")
        box.setAttribute("aria-describedby", "basic-addon2")
        box.setAttribute("id","input"+str)

        document.getElementById("element"+str).appendChild(box)
    }
        
    d3.select(".btn.btn-success.nextBtn").on("click", function () {
        var endTime = new Date().getTime();
        var timeSpent = endTime - startTime;
        ratings = []
        for(var i = 0; i < frames; i++){
            var str = i.toString();
            rating = document.getElementById("input"+str).value
            ratings.push(rating)
        }
        sendData(id, userID, timeSpent, ratings)
    })
}

function sendData(id, userID, time, ratings){
    console.log("sending data")
    
    //url2go =  id + "/rankings"
    url2go = userID + "/" + id + "/B/sendRatings/"

    data2send = [time, ratings]
    
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

}