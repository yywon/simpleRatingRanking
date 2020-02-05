    #NOTE: Use this code in future
    
    rank_dif = None
	rate_dif = None
	rank_ui = None
	rate_ui = None	

	if user["surveyResults"] is not None:
		item = user["surveyResults"]
		print(item)
		ranking_mental_demand, rating_mental_demand, ranking_performance, rating_performance = fixResults(item, ranking_mental_demand, rating_mental_demand, ranking_performance, rating_performance)