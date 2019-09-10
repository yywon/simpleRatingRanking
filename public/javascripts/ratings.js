allquestions = [
    [6, [200, 206, 212, 218]],
    [10, [200, 210, 220, 230]],
    [14, [200, 214, 228, 242]],
    [18, [200, 218, 236, 254]]
]

renderQuestion(allquestions[3], 0);

function renderQuestion(question, index){

    groupnum = question[0]
    candidates = question[1]
    candidate = candidates[index]

    d3.select(".activity").html("")

    d3.select(".activitypage .activity").append('div').attr("class", "image")
    d3.select(".activitypage .activity").append('div').attr("class", "slider")

    let svgimagewidth = "100%"
    let divWidth = d3.select('.activity').node().offsetWidth
    let imageSize = divWidth / 3
    let svgimageheight = imageSize + 50;

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

    

        

            


    
    
    
    

        
        
    









}