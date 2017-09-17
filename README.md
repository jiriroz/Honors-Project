#Notes
* Account for unusual events? http://gothamist.com/2015/02/26/jfk_runway_bye_bye.php
* Compute delay per airport separately or make airport be one of the variables? Same with airline
* Which years to use? Is performance consistent year to year, month to month?
* Database of flights https://www.transtats.bts.gov/TableInfo.asp
* Database of delay reasons https://www.rita.dot.gov/bts/help/aviation/html/understanding.html
* FAA airport-specific data: https://aspm.faa.gov/
* 538: https://fivethirtyeight.com/features/how-we-found-the-fastest-flights/
* Overview of approaches: https://arxiv.org/pdf/1703.06118.pdf
* Analysis: http://ddowey.github.io/cs109-Final-Project/#!index.md
* Which resources are scarce/critical? Airplane, cockpit, cabin crew
* Do airplanes wait for connecting passengers?
* Paper on delay propagation: http://web.mit.edu/sis07/www/cohn.pdf
* Use both delay prediction and propagation: http://www.sciencedirect.com/science/article/pii/S0968090X14001041
* Dataset of flight routes: https://openflights.org/data.html
* Need to find database of flight schedules and real-time updates API
* Cancelled flights: Passengers need to be put on the next flight, which causes the passengers of that flight be put on the next one and so on. Can we simulate this? How to account for cancelled flights in general? (look at 538 analysis)
* Should we predict delay events independently or together as a single event?

#Possible features
* Month of year
* Day of week
* Time of day (hour) of departure and arrival
* Duration of flight (turns out longer flights are less likely to be delayed since they have higher priority)
* Departure and arrival airport
* Airline
* Demand at the airport
* Weather at the airport - temperature, visibility, others
* Think of non-traditional features/data sources (Twitter)
* Congestion at an airport

Some features may be correlated (airport and weather)

#Network model
* Simulate every flight or only the queried one? (depends on whether they’re dependent)




#Project Description
* Create a model to predict delays for near-future commercial flights in the U.S. More specifically:
* Given a flight, estimate the duration of its delay
* Will be focusing on near-future flights (several days into the future). This is because we can get weather forecast. However, potentially allow capability to predict for long-term future flights with higher uncertainty.
* Will be using roughly this set of features: Airport, airline, weather data (visibility, temperature, etc.), month, day of the week, hour, duration of flight, demand, congestion.
* These features will be used in a statistical model that will predict duration of delay for a single flight.
* Furthermore, many delays are caused by critical resources (airplane and crew) being delayed in the previous flight. The delay then propagates to further flights. For this reason I will build a network model of airports and flights, where I will model delay propagation.
* The resulting model will incorporate both the network model and the statistical one.
* Finally, build a website/app allowing users to enter a flight and get the expected delay.


#Project Plan
* Identify data sources
..* For flight delay prediction - historical about flights, delays, and weather
..* For delay propagation - flight routes
..* For real-time prediction - flight schedules, tracking, weather forecasts
* Create a model for delay prediction for a single flight
..* In a more abstract way, we are predicting occurrences of several different kinds of events. Each kind of event has a different probability of occurrence and causes a delay of different duration.
..* Get, process the data.
..* Perform exploratory data analysis
..* Experiment with different kinds of models and see which one works best.
* Create a model for delay propagation
* Combine delay prediction and propagation
..* Predict delay events happening and simulate their propagation throughout the flight network.
* Implement real-time version of the model
..* Get the latest flight schedule and flight tracking data
..* Get the weather forecast data
..* Identify other real-time data sources I could use.
..* Using these, simulate our model and predict flight delays several days into the future
* Create a website/app that allows users to use our model interactively
..* The interface will allow users to enter a flight, and it will display the predicted delay.


