# SVD Energy Day-Ahead Prices

## Motivation & Overview
My curiosity in applying knowledge from different fields drove the creation of this project. The tools I utilized come from Linear Algebra, Macroeconomics, Optimization, Econometrics, Data Science and Numerical Mathematics.

I chose to analyze energy market data, because of its nice interpretations. Electricity cannot be easily stored on a large scale. This physical limitation creates an extremely volatile day-ahead market that reacts violently to daily human behavior and weather patterns. Tracking how solar generation and morning consumption spikes shift the merit order curve provides a perfect environment to test statistical models.

This project analyzes historical hourly day-ahead electricity prices in France for the year 2022. The analysis extracts market risk factors from the price volatility using SVD. By approximating the original data matrix with a lower-rank representation, the model eliminates unpredictable market noise. The resulting principal components dictate the geometric shapes of independent market movements.

The output is a target curve for hedging energy generation profiles. Through the method of least squares optimization, the model calculates the market exposures needed to neutralize an asset's risk. Regressing the production vector onto the principal components calculates overlaps and measures how much generation happens during the hours that an economic shock shifts the energy market. This provides a blueprint to assemble an optimal portfolio of standard exchange products.

## Data Credit
The dataset containing the hourly European day-ahead electricity prices was sourced from Henri Upton, github.com/henriupton99.