function renderQuestion(question, picture, id, userID){

    var startTime = new Date().getTime(); 

    groupnum = question[0]
    candidates = question[1]
    candidate = candidates[picture]

    d3.select(".activity").html("")

    d3.select(".activitypage .activity").append('div').attr("class", "image")
    d3.select(".activitypage .activity").append('div').attr("class", "slider")

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
            .attr('xlink:href', "/images/dots/" + groupnum + "/" + candidate + ".png")
            .attr("x", imageSize + 2)
            .attr("y", 2)
            .attr("width", imageSize)
            .attr("height", imageSize)
    
    var data = [0, 50, 100, 150, 200, 250, 300, 350, 400];

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


/*
    var sliderSimple = d3
        .sliderBottom() 
        .min(d3.min(data))
        .max(d3.max(data))
        .step(1)
        .width(500)
        .tickFormat(d3.format(''))
        .ticks(5)
        .default(50)
        .on('onchange', val => {
            d3.select('p#value-simple').text(d3.format('')(val));
        });
          
    var gSimple = d3
        .select('div#slider-simple')
        .append('svg')
        .attr('width', 700)
        .attr('height', 70)
        .append('g')
        .attr('transform', 'translate(100,30)');
          
    gSimple.call(sliderSimple);
          
    d3.select('p#value-simple').text(d3.format('')(sliderSimple.value()));

*/


    

        

            


    
    
    
    

        
        
    



