# Time-series-data-visualization-to-recognise-past-performance-and-forecast-game-outcome-
Application of time series data visualization to recognise past performance and forecast game outcome at the English Premier League

**Background**
In recent years, there has been a growing amount of study focused on assessing the technical and tactical performance of teams in association football using data from match events. Nevertheless, the majority of research employed a unidimensional methodology and examined the impact of individual performance factors on match outcomes separately, hence restricting the comprehensibility of the findings. The objective of the study was to utilise an advanced algorithm to rank team performance and analyse key performance indicators in relation to match outcomes using a large dataset of matches. 

Traditionally, 20 teams compete in the English Premier League (EPL), and each club play each of the other 19 teams twice per season, making a total of 38 games for each team and 760 games per season. In an attempt to forecast the outcome of a match involving the home and away teams, each of the models is given consideration, leveraging on the mid-point of the three-class model, which means picking draw in the (win, draw, loss) range. In order to determine the accuracy of the predicted values, several rounding procedures were applied to them. After comparing the results, it was determined that splitting the training and test data by 70% and 30% respectively

First, the study adopted the logistic regression and decision on tree in the first layer of the dataset (2000 – 2018) and then applied linear regression, decision tree and neural network (as a form of LSTM) on two season-length data (2020 – 2021 and 2021 – 2022) as a basis for predicting football match outcome. Then, the study thoroughly studied the time aspect of the soccer match result of the EPL soccer league and introduced three separate models to predict the future game result. All three models could predict the result with a 70%+ prediction accuracy. However, soccer is a highly random game and depend on numerous factors like players emotions; 70% + accuracy is quite high to predict a game. On the flip side, improving the LSTM model by modifying the model parameters and adding more layers to the model might improve the prediction result.

 The findings indicate that the decision tree model performed bettern than other models and yielded a higher training accuracy of 97.98% in the full dataset (2000 – 2018) when compared to the decision forest score (R2) of 0.99 in 2020-2021 and 2021-2022 seasons. Despite the superiority of the decision tree model, which is known to be prone to overfitting, the simplistic assumption of linear decision boundaries, the logistic Regression algorithm can provide an advantage, especially when dealing with larger amount of data for prediction.


**Project Aims and Objectives**
To forecast future game outcome based on identified variables that represents past performances (see rationale).
The specific objectives are:
i.	To explore whether or not football game outcomes can be predicted, 
ii.	To understand the significance of match statistic input variables and their role in game outcome prediction, and 
iii.	To review the contribution of past authors on the analysis of player and team performance
iv.	To assess the value of select machine learning models in predicting game outcome based on past performances, adjusted for the emotional (non-input) factors.
v.	In light of the limitation gaps, proffer suggestions based on the comparative performance of the machine learning methods.



**Project Specification**
Information about the dataset, the machine learning methods used, and the metrics used to evaluate the models' efficacy are presented here. This study used Python to carry out the implementations similar to Rahman et al. (2018) and Razali et al. (2017). Time series prediction of the average score of upcoming football matches is performed using a sliding window and LSTM model. The KDD approach (Knowledge Discovery in Databases) is used to analyse the data. Figure 1 depicts the stages of the research.

Figure 1: Model Framework

![image](https://github.com/user-attachments/assets/e877a1ab-8ca1-4c4b-8f84-9c303034f1a2)

**Design Methodology**
Bayesian algorithms such as Naive Bayes and Tree-Based Naive Bayes for comparing the outcomes, as well as an LSTM (Long Short-Term Memory) Sequential model for making time series predictions, were utilised in this investigation as the methodologies for doing the research. In an attempt to forecast the outcome of a match involving the home and away teams, each of the models is given consideration, leveraging on the mid-point of the three-class model, which means picking draw in the (win, draw, loss) range.


Figure 2. Design of Time series-based Models

![image](https://github.com/user-attachments/assets/d01da6a1-c70a-4b94-bcb5-39f7580421eb)


**Dataset**
The availability of an appropriate dataset is crucial for football match prediction. Due to the nature of the game's volatility, every available statistic must be used in the analysis. For this analysis, Premier League game results from the 2000-2001 to 2017-2018 season was used. In addition, the 2020–2021 and 2021 – 2022 season were used for comparative analysis. 

Traditionally, 20 teams compete in the English Premier League, and each club play each of the other 19 teams twice per season, making a total of 38 games for each team and 760 games per season. The datasets are collected from Kaggle (https://www.kaggle.com/saife245/english-premier-league), a subsidiary of Google LLC, and an online platform for data scientists and machine learning practitioners

This means there are 380 games in a season. Each downloadable CSV file includes 380 rows of match information such as participating teams, date, individual game statistics, and a few betting amounts that are unnecessary for the analysis. A three year the same three-year dataset (2014/2015, 2015/16, and 2016/17) was used similar to Rahman et al. (2018)’s implementation. In their analysis, they factored in data from three separate seasons: 2014–2015, 2015–2016, and 2016–2017.


**Data Pre-processing**
The dataset that was obtained is a collection of CSV files that are organized seasonally and contain information on all of the games that took place during that season. Each file has a total of 380 rows, but only 68 out of 20 columns are really being used for the research. Google Collaboratory Python version was used for tasks involving the cleaning and transformation of the dataset. Python libraries such as "pandas," "NumPy," "matplotlib," and "glob" are loaded when the program is first run.

In order to retrieve the rows included within each file, the 'pandas' read csv function, in conjunction with the glob library, is utilised. Because the supplied data contained a large number of columns that were not wanted, pandas was used to eliminate those columns. The dataset is examined for any instances of null values or values that are not expected. Only the date, home team, away team, and FTR (Full Time Result) are picked for the time series, however all of the game statistics and team details columns are used for the Bayesian methods. 

For the purpose of the time series analysis, the values of the column FTR are either A (representing the away team), D (representing a draw), or H (representing the home team). These values are then converted to 1, 0, and -1, respectively. Teams that have participated in the English Premier League for a relatively small number of seasons will not be included in the time series projection.

**Feature Justification**
Host team: This is the home team. Hosting give the club advantage of not having to travel, a familiar playground, and of course, fan encouragement.
 
HTP, ATP: HTP (Home Team Point) and APT (Away Team Point) indicate the average point that the team earns in that season. This is a good reflection of the corresponding position between the two teams at the moment.
HTGD, ATGD: HTGD (Home Team Goal Difference) and ATDG (Away Team Goal Difference) is the average of the goal the team scored and conceded in that season. In football, the team that scores more goals is the winner; therefore, I believe these two features are strong indicators of the team’s winning probability.


DiffPts: Difference of current season points between two teams (home’s – away’s): The team result can vary significantly among different seasons due to reasons such as changes in line-up. It’s important to take into account how the team are performing in this season to predict their future matches.

DiffFormpPTS (H1 – H5, A1 – A5): Result of the last 5 matches of the team. This will help us know how the team is performing recently.
Furthermore, the data was visualised by using the scatterplot and all of the features show a strong correlation with the data for predict – the ‘Result’ (see appendix A for a compendium of the variable definition used in the dataset for 2020-2022.



**Implementation**
To forecast future game outcome based on identified variables that represents past performances (see rationale). More specifically, learn whether or not football game outcomes can be predicted to understand the cultural significance of certain input variables and their role in game outcome prediction, and in general assess the value of select machine learning models in predicting game outcomes based on past performances, adjusted for team psychology.
1.	Regression (logistic-based) and decision tree were used to validate the full dataset (2000 – 2008), providing a basis for how football game outcomes can be predicted.
2.	Regression (linear-based), decision tree, and recurrent neural network (novel LSTM)were used to assess the value of machine learning in predicting the next game outcome based on previous match events.





