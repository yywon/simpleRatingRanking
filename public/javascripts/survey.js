Survey
    .StylesManager
    .applyTheme("default");
let demJson = {
    questions: [
        {
            name: "age",
            type: "text",
            title: "**What is your age** ?",
            placeHolder: "",
            isRequired: true
        },
        {
            type: "radiogroup", hasOther: false, isRequired: true, name: "gender", colCount: 1,
            title: "**What is your gender** ?",
            choices: [
                { value: "Male", text: "*Male*" },
                { value: "Female", text: "*Female*" },
            ]
        },
        {
            type: "radiogroup", hasOther: false, isRequired: true, name: "education", colCount: 1,
            title: "**What is your current level of education** ?",
            choices: [
                { value: "Less than High School", text: "*Less than High School*" },
                { value: "High School/GED", text: "*High School/GED*" },
                { value: "Some College", text: "*Some College*" },
                { value: "2 year degree", text: "*2 year degree*" },
                { value: "4 year degree", text: "*4 year degree*" },
                { value: "Master's", text: "*Master's*" },
                { value: "Doctoral", text: "*Doctoral*" },
                { value: "Professional (MD, JD, etc.)", text: "*Professional (MD, JD, etc.)*" },
            ]
        },
        {
            name: "major",
            type: "text",
            title: "**If you have been or are enrolled in a post high school institution, what is your major** ?",
            placeHolder: "",
            isRequired: true
        },
        {
            type: "radiogroup", hasOther: false, isRequired: true, name: "employed", colCount: 1,
            title: "**Are you currently employed** ?",
            choices: [
                { value: "Yes", text: "*Yes*" },
                { value: "No", text: "*No*" },
            ]
        },
        {
            name: "job",
            type: "text",
            title: "**If yes to #5, what is your job title** ?",
            placeHolder: "",
            isRequired: true
        },
        {
            type: "radiogroup", hasOther: true, isRequired: true, name: "nativeSpeaker", colCount: 1,
            title: "**Are you a native English speaker** ?",
            otherText: 'If No, then what is your native language?',
            choices: [
                { value: "Yes", text: "*Yes*" },
                // { value: "No", text: "*No*" },
            ]
        },
        {
            type: "radiogroup", hasOther: false, isRequired: true, name: "stayInUS", colCount: 1,
            title: "**How long have you lived in the United States** ?",
            choices: [
                { value: "Native (all my life)", text: "*Native (all my life)*" },
                { value: "Less than 1 year", text: "*Less than 1 year*" },
                { value: "1 year", text: "*1 year*" },
                { value: "2 years", text: "*2 years*" },
                { value: "3 years", text: "*3 years*" },
                { value: "4 years", text: "*4 years*" },
                { value: "Greater than 5 years", text: "*Greater than 5 years*" },
            ]
        },
        {
            type: "radiogroup", hasOther: false, isRequired: true, name: "Heatmaps", colCount: 1,
            title: "**Have you heard about Heatmaps, as the concept we used in the experiment before** ?",
            choices: [
                { value: "Yes", text: "*Yes*" },
                { value: "No", text: "*No*" },
            ]
        },
    ],
    // completedHtml: "**Thank you for completing the survey. Please click the 'Finish' button to get your key!**"
}

function startFromSurvry() {
    ////show demgrophyPage
    ////empty the page
    $(".demograpyPage").show()
    Survey.defaultBootstrapCss.navigationButton = "btn btn-green";
    Survey.Survey.cssType = "bootstrap";

    let survey = new Survey.Model(demJson);

    survey.onComplete.add(function (result) {
        ////show the surveyPage to show the key
        let userDemographic = result.data
        console.log("userDemographic ", userDemographic)
    
        /*
        let key2payMoney = commnonFun.key2pay()
        // console.log("key2payMoney ", key2payMoney)
        commnonFun.callRouter({
            reqType: "saveUniqKey2pay",
            userid: userid,
            key2payMoney: key2payMoney,
            userDemographic: userDemographic
        }).then(d => {
            console.log("the result of ", d)
            d3.select(".surveyPage").append("div")
                .attr("class", "userName")
                .html("Welcome: " + userid)
            d3.select(".surveyPage").append("div")
                .attr("class", "endpage")
                .html("Thank you for participating our user study. Please fill the following code in \
    Amazon Mechanical Turk to get your compensation. <p>" + key2payMoney + "</p>")
        })
        */
    });

    let converter = new showdown.Converter();
    survey.onTextMarkdown.add(function (survey, options) {
        //convert the mardown text to html
        let str = converter.makeHtml(options.text);
        //remove root paragraphs <p></p>
        str = str.substring(3);
        str = str.substring(0, str.length - 4);
        //set html
        options.html = str;
    });

    $("#surveyElement").Survey({
        model: survey
    });

}