function renderQuestion(question, picture, id, userID){

    var startTime = new Date().getTime(); 

    d3.select(".activity").html("")

    d3.select(".activitypage .activity").append('div').attr("class", "image")

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
    let divWidth = d3.select('.activity').node().offsetWidth
    let imageSize = divWidth / 3
    let svgimageheight = imageSize+ 20;

    //grouping for image
    let svg4image = d3.select('.image').append('svg')
        .attr("width", svgimagewidth)
        .attr("height", svgimageheight)

        let g4rankingborder = svg4image.append("g")
        .attr("class", "g4rankingborder")
        .selectAll("rect")
        .data([1])
        .enter().append("rect")
        .attr("x", imageSize)
        .attr("y", 0)
        .attr("width", imageSize + 4)
        .attr("height", imageSize + 4)
        .attr("style", function (d, i) {
            return "fill:transparent;stroke:black;stroke-width:1;stroke-opacity:1"
        })

        let g4image = svg4image.append("g")
            .attr("class", "g4image")

        let img = g4image.append("image")
            .attr('xlink:href', "/images/dots/" + candidate + ".png")
            .attr("x", imageSize + 2)
            .attr("y", 2)
            .attr("width", imageSize)
            .attr("height", imageSize)

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