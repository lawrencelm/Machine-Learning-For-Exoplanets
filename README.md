# Machine-Learning-For-Exoplanets

Open-Ended Project: The Search for Humanity’s Next Home

by Albert Wandui and Lawrence Lin Murata
Spring 2015

“Mankind was born on Earth. It was never meant to die here.” 
 — Cooper from Interstellar (2014) [3]

	1. The “Finding Home” Project

Our project aims at finding exoplanets that are habitable to human beings using machine learning and astrophysics. An exoplanet is is a planet that orbits a star other than the Sun, a stellar remnant, or a brown dwarf. [6] We used data sets from NASA to study how properties that would, in theory, be relevant to the habitability of an exoplanet by humans can help us predict whether certain exoplanets we know of would be habitable or not.

Using our astrophysics knowledge to select the properties, the best data and use the available data in the most relevant way, we train two variations of Naive Bayes (Maximum Likelihood Estimate and Laplace Estimate) and Logistic Regression to classify the data and predict whether each planet is habitable or not (Y = 0 for not habitable and Y = 1 for habitable).

A - Our results
	
	Naive Bayes:

Using Maximum Likelihood Estimator

Class 0: tested 798, correctly classified 775
Class 1: tested 71, correctly classified 67
Accuracy = 0.97

Using Laplace Estimator

Class 0: tested 798, correctly classified 775
Class 1: tested 71, correctly classified 67
Accuracy = 0.97

Logistic Regression:

Class 0: tested 798, correctly classified 775
Class 1: tested 71, correctly classified 67
Accuracy = 0.97

	It was interesting to see how accurate Naive Bayes (with both Laplace and MLE) and Logistic Regression were in predicting whether an exoplanet is potentially habitable or not, even though we had to simplify some factors in our data, as described later in part 2 “Data Set”. We go into more detail about the significance of the results observed and what we learned from them in part 3 “Significance.” 

	B - Some of the libraries, tools and knowledge required

numpy
python library that helped us model and use the data
Astrophysics knowledge about exoplanets and about habitability of exoplanets
learned through PHYSICS 100: Introduction to Observational and Laboratory Astronomy
and by reading many papers on habitability, exoplanets and astrophysics
Machine Learning
Naive Bayes
Laplace Estimate
Maximum Likelihood Estimate
Logistic Regression
Classifier
Modelling the data
making sure all data is accurate, binary and relevant to our project


	2. Data Set
	
	Our data set comes from the NASA Exoplanet Archive, made available thanks to NASA Exoplanet Institute, the Infrared Processing and Analysis Center and Caltech. [4] We chose the NASA data set because it is the most comprehensive and detailed archive for exoplanets, with over 100 properties of 1852 confirmed exoplanets (over 7000 unconfirmed and confirmed exoplanets) available. The data set about habitable planets is from the Planetary Habitability Laboratory.

	One of the lists we researched was a list of potentially habitable planets made available by the Planetary Habitability Laboratory [5]. The methods used by the laboratory to find the most potentially habitable exoplanets is rigorous and includes factors like ESI (Earth Similarity Index), HZD (Habitable Zones Distance), HZC (Habitable Zone Composition), HZA (Habitable Zone Atmosphere), SPH (Standard Primary Habitability), pClass (Planetary Class) and hClass (Habitable Class) [8].

	The other list researched was the list of not habitable planets. It required much more work than predicted (predicted, get it?). The reason for that is that people are generally more curious about knowing which planets are habitable for humanity rather than which planets are not habitable. Thus, we had to generate our own data for this category, calculating an estimate of each exoplanet’s ESI (Earth Similarity Index) [9] and also its HZD (Habitable Zone Distance), which were used as some of the main factors to determine which planets had the most extreme conditions and were not habitable at all. The HZD of a planet depends mostly on the mass and radius of the host star (the system’s star), the star’s radiation and between the distance and the planet to the star [7].


	A - Modelling the data
	
	A lot of the work required in this project involved understanding the data available, selecting the relevant information using our knowledge of Physics and astronomy, and being able to format the data sets in a way that allows us to properly use our machine learning algorithms on them. We also had to convert features to binary values, partition the data into test and train, and decide how to work with each feature.

Binary data:

We had to turn all the information available about the selected properties. First, we encoded the predicted value as Y=0 for exoplanets that are not habitable and Y=1 for habitable exoplanets.
For the properties/input variables we were analyzing, we chose Xi=0 if the measurement of the property was lower than Earth’s and Xi=1 if it was higher. We describe which properties we selected as input variables and why in the “B - Properties selected” section under item “3. Significance” 

Choosing the types of planets:

	We had to make sure we were using planets that could meet our criteria, which included being a confirmed planet and having enough available information for us to calculate the ESI (Earth Similarity Index) and to compare the data among planets from the two different data sets (NASA Exoplanet Archive and Planetary Habitability Laboratory).

Formatting the data:

	We wrote code (available in section “Code for Modelling Data” in the end of the document) to format the data following these steps:

Load the data into python from 2 distinct data sets. One from exodata.csv which contains the habitable planets and one from exodata1.csv which has non-habitable planets.
Convert all the data into binary values by comparing them against the earth values. Now we can cast the new boolean values into 1 for True and 0 for False. The names allow us to directly access the columns we want in the structured array.
Make the training data set by combining half the data from habitable_file and 200 random non-habitable exoplanets from the non-habitable data set. All variables should be binary. The ESI is the measure of similarity of a planet to Earth. For our calculations, all planets in the habitable data set of habitable planets have ESI of 1.
Combine all the data into one train data array and create the test data array.
Write all the data into txt files.


	3. Significance

	A - Results Observed

Simply from the statistical distributions of exoplanets, we expect to have relatively few potentially habitable planets. The success of the classification thus shows that there any a number of key parameters that can be successfully used as predictors even in models as simplified as this. A routine such as this can help an astronomer or planetary scientist narrow down potential target planets and focus on selecting exoplanets that meet some specific criteria of interest such as habitability.

	B - Properties selected

	We selected relevant properties from our data set based on Planet Habitability and aspects we thought would be relevant for habitability. Another factor considered was whether the information about the properties was available in both major data sets (NASA Exoplanet Archive and Planetary Habitability Laboratory) we were working with. The following properties are the ones selected:

Radius of exoplanet: The radius of the planet is easily measurable for exoplanets discovered through transits. The radius of the planet (as well as the mass) affect the density of the planet which determines whether a planet is rocky like Earth and can possibly support life or a gas giant planet like Jupiter which doesn’t support life.
Insolation Flux: This is the amount of radiation that the planet receives from its host star. Given that the sun is the chief source of energy for life on Earth, the amount of flux that a planet receives influences its capacity for hosting life.
Equilibrium Temperature: Most life on Earth can only survive and thrive when temperatures lie within a narrow range. If a planet has Earth-like temperatures on average, then it may be a good candidate for life forms similar to those seen on Earth. Most exoplanets however, are vastly dissimilar from Earth in this respect. Perhaps, then life similar to that found in extreme environments such as geothermal vents may thrive in such vastly different conditions.
Period: Kepler’s Third Law relates the period of orbit of a planet to its average distance from the star. T2 = k . a3 where T is the period, k is a constant and a is the average distance. Planets that have very short periods are very close to their host star and are constantly baked by radiation which is unconducive for life (think Mercury). Planets too far away, don’t receive enough radiation hence are also unsuitable for life. Only Goldilock planets where the distance is just right can support life similar to what is seen here on Earth.

C - Humanity’s Future and Survival: searching for our next home

	The question we answered by predicting the potential habitability for each exoplanet is one that is relevant to all of us humans: “can this planet potentially become our new home?” 
 
Unfortunately, we have been spending the resources Earth has in a prejudicial way. Since the industrial revolution, the rate at which humanity consumes resources can been increasing astoundingly year over year, making many of us worry if the next generations will have the resources to live properly. [2]

	At the same time, NASA and other space traveling research have received considerable budget cuts in the past few years due to a combination of shift in focus for the government, lack of optimism and few results. [1]

	In times like these, it is crucial to answer the question of whether we can habitate in exoplanets or not, how far they are and how many potentially habitable exoplanets are out there. By using this data the way did in order to predict the habitability of all the known exoplanets we can help not only astrophysicists and aero engineers answer these questions but we can also find out relevant statistics that quantify questions that have fueled human imagination for thousands of years. The search for the unknown and the question of whether we can live in a planet or not have inspired astronomers in Ancient Greek to spend their lives studying the sky or inspired directors to create mind-blowing movies like Kubrick’s 2001: Space Odyssey or Nolan’s Interstellar.

We want to make sure humanity will be able to thrive and survive despite all the limitations faced on Earth and, to do so, we need to first know whether we can habitate othe planets and where/which these planets are. Machine learning, used together with astronomy, can help us search for humanity’s next home.

“We've always defined ourselves by the ability to overcome the impossible. And we count these moments. These moments when we dare to aim higher, to break barriers, to reach for the stars, to make the unknown known. We count these moments as our proudest achievements. But we lost all that. Or perhaps we've just forgotten that we are still pioneers. And we've barely begun. And that our greatest accomplishments cannot be behind us, because our destiny lies above us.”  
— Cooper from Interstellar (2014)

Bibliography:

[1] Nasa budgets: US spending on space travel since 1958 UPDATED
http://www.theguardian.com/news/datablog/2010/feb/01/nasa-budgets-us-spending-space-travel 
[2] Humankind/nature Interaction: Past, Present and Future
https://books.google.com/books?id=s5loBk4dMzsC&pg=PA40&dq=humans+destroying+nature&hl=en&sa=X&ei=PA11VdWoCcjWoATH-IPACQ&ved=0CCMQ6AEwAQ#v=onepage&q=humans%20destroying%20nature&f=false 
[3] Nolan, Christopher. Interstellar (2006)
[4] NASA Exoplanet Archive http://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=planets 
[5] Potentially Habitable Exoplanets http://phl.upr.edu/projects/habitable-exoplanets-catalog 
[6] Schneider, J. "Interactive Extra-solar Planets Catalog".The Extrasolar Planets Encyclopedia.
[7] What’s New in the NASA Kepler Data http://phl.upr.edu/press-releases/whatsnewinthenasakeplerdata 
[8] What Makes a World Habitable? http://www.lpi.usra.edu/education/explore/our_place/hab_ref_table.pdf 
[9] Earth Similarity Index (ESI) http://phl.upr.edu/projects/earth-similarity-index-esi 




