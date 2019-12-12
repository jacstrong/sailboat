# Implement Simulated Annealing
## Jacob Strong

I tried to answer how is run time affected as the number of cities grows? Also how was output quality affected.

I ran six experiments each with 20 trials. Each trial tracked the initial distance, the final distance and the number of iterations.

I found that increasing time was roughly linear. With 10 cities taking less than 10 seconds, 100 cities taking roughly a minute, and 1000 cities taking several minutes. I think that this would hold true unless your temperature reduction became dependent on the number of cities.

Because for each set we do not know the best distance, I measured the improvement from the initial randomized route and the final optomized route. With smaller numbers of cities the improvement was always higher. I imagine that this is because my temperature function had nothing to do with the number of cities. If the temperature were changed to be dependent on the number of cities I am sure that could be changed.

There were some problems with my algorithm. There were cases where the final result would be greater that the initial result. I can only imagine this is due to probability and bad solutions getting accepted. I did make a change to the way my temperature was calculated and found that in almost all cases it eliminated this problem.

See data.csv for my trial results.
