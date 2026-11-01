# Next-try pricing in the NRL

This repository tests whether a major online bookmaker's live "next try" odds undervalued the team that had just scored. NRL games from 2013 to 2019 say they did. Applied to the odds actually taken, the scoring rates from those games predict a flat ROI of +9.7%, and a season of real trades returned +11.7%.

**Summary**

| | |
|---|---|
| Trades | 703, NRL Premiership, 2026 season |
| Implied probability (1 / odds) | 48.2% |
| 2013-2019 lookup table probability | 52.2% |
| Predicted ROI from the lookup table | +9.7% (95% CI +7.5% to +12.0%) |
| Actual flat ROI | +11.7% (95% CI +3.7% to +19.8%) |

**The market**

After each try the bookmaker offers odds on which team will score the next one. If no further try is scored, the bet is void. Every trade here was placed after at least one try had been scored.

**The edge in the historical data**

Every NRL game from 2013 to 2025 was collected from nrl.com, along with each game's pre-match odds. For each try in 2013-2019 that was followed by another try, the table below shows how often the same team scored the next one. The 8,403 moments are grouped by the scoring team's pre-match win probability (from the odds, with the margin removed) and its lead after the try. The pre-match probability only sorts the moments into groups. The probability of scoring next is counted from what actually happened.

| Pre-match win probability | Trailing or level | Leads by 1-6 | Leads by 7-12 | Leads by 13+ |
|---|---|---|---|---|
| Under 35% | 45% | 39% | 39% | 37% |
| 35-50% | 51% | 48% | 46% | 52% |
| 50-65% | 60% | 55% | 56% | 55% |
| 65-80% | 63% | 60% | 58% | 68% |
| Over 80% | 73% | 69% | 67% | 79% |

Each trade is assigned the probability p from its cell, or 1 - p if it backed the other team. Its expected return is p × odds - 1, and the mean over the 703 trades is the predicted ROI, +9.7%. The calculation uses each trade's odds but not its result. Rebuilding the table from 1,000 bootstrap resamples of the 2013-2019 matches gives a 95% CI of +7.5% to +12.0%, and no resample predicts an ROI at or below zero.

<img src="figures/1-price-vs-history.png" width="620" alt="Lookup table probability against implied probability for the 703 trades in seven groups. Every group sits above the fair-odds line.">

**The live test**

The trades were placed through the 2026 season, starting with small stakes that grew as the results held up. Their results were never used in the prediction above, so they are an independent test of it.

| | |
|---|---|
| Implied probability | 48.2% |
| Win rate | 53.3% |
| Flat ROI | +11.7% |
| Stake-weighted ROI | +11.3% |

Across every trade in the file (783 in total, including NRLW, the World Cup and State of Origin), the total staked was $420,175 for a profit of $47,902, a flat ROI of 11.6%.

![Predicted ROI of +9.7% (95% CI +7.5% to +12.0%) beside the actual ROI of +11.7% (95% CI +3.7% to +19.8%).](figures/2-predicted-vs-actual.png)

![Cumulative profit through the 2026 season. Actual profit finished at $41,665 against $30,626 expected from the 2013-2019 lookup table.](figures/3-profit-over-season.png)

**Statistical significance**

Trades in the same game aren't independent, so the bootstrap resamples whole games, 20,000 times. The 95% CI for flat ROI is +3.7% to +19.8%, and P(ROI ≤ 0) = 0.002. Treating every trade on the same day as one cluster gives almost the same interval, +3.6% to +19.7%. The method was also checked on simulated trades with no edge, each winning with probability exactly 1 / odds. Across 1,000 simulated datasets, the 95% CI contained zero between 93% and 97% of the time, so the bootstrap isn't overconfident.

![Bootstrap distribution of flat ROI from 20,000 resamples of whole games, centred near +11.7%, with P(ROI ≤ 0) = 0.002.](figures/4-statistical-significance.png)

**The theory that didn't hold up**

The original theory was that the 2020 "six again" rules were the reason, because a faster game should mean more runs of tries. It has been evident the NRL's interim CEO, Peter V'landys, is inclined to introduce a new 'kickoff' rule (letting the opposing team choose whether to receive or kickoff after a try is scored), in an attempt to slow down 'momentum'. This rule was tested in 2 dud games towards the end of the 2026 NRL season. These actions only reinforce the curiosity that consecutive runs of momentum were happening more often, hence statistical tests were conducted. The test compares the real order of tries in each game with the same tries in a random order. Teams do score back-to-back more often than chance, but no more since the rule change than before.

| | 2013-2019 | 2021-2025 |
|---|---|---|
| Tries per game | 7.0 | 7.9 |
| Same team scored the next try | 54.8% | 55.1% |
| Excess over random order | +2.7% (95% CI +1.8% to +3.6%) | +2.2% (95% CI +1.2% to +3.2%) |

The excess is the share of consecutive tries scored by the same team beyond what each game's try counts would produce in a random order, so it isn't inflated by strong teams scoring more tries. A permutation test with 10,000 shuffles puts the 2021-2025 excess well above zero (p = 0.0001). The change after six-again is -0.5 points (95% CI -1.9 to +0.8). The rules did bring more tries per game, which meant more chances to bet, but the edge was already there before them.

![Tries per game by season, rising after 2020, beside the share of next tries scored by the same team, steady at about 55%.](figures/5-rule-change.png)

**Caveats**

The odds are only known for trades that were actually placed, because there's no public record of live next-try odds. So this shows the odds were too long on these trades, not on every moment after a try. The lookup table only knows each team's pre-match strength and the score. It can't see what happens during a game, such as injuries, sin bins or which side is on top. The table's groups were set once, at round numbers, and not adjusted afterwards. It's one season of trades, and the implied probability includes the bookmaker's margin, so the bookmaker's own estimate was lower still.

**Data**

Match data comes from nrl.com's match centre and pre-match odds from [AusSportsBetting](https://www.aussportsbetting.com/data/). Neither is included here. The odds file is free for personal use, and the match data is rebuilt by the scraper. data/trades.csv has every trade, with its time, competition, team, try number, odds, stake and result. Its match IDs number the games in the order they were traded.

**Repository layout**

| File | Purpose |
|---|---|
| data/trades.csv | Every trade |
| src/scrape.py | Downloads finished matches from nrl.com |
| src/nrl_microstructure/parse.py | Turns the saved matches into match and try tables |
| src/nrl_microstructure/odds.py | Pre-match win probabilities from historical odds |
| src/nrl_microstructure/join.py | Links each trade to its match and the tries either side of it |
| src/nrl_microstructure/evaluate.py | Flat and stake-weighted ROI and the bootstrap |
| src/nrl_microstructure/history.py | The 2013-2019 lookup table, the predicted ROI and the rule-change test |
| src/nrl_microstructure/clustering.py | The shuffle test for back-to-back tries |
| src/nrl_microstructure/figures.py | Draws the figures |
| tests/ | Tests for each step |

**How to run it**

The live test needs only data/trades.csv.

```
pip install -e ".[dev]"
python -m nrl_microstructure.evaluate
pytest
```

Rebuilding the historical data takes two to three hours, mostly to collect every game from nrl.com.

```
python src/scrape.py --seasons 2013-2026
python -m nrl_microstructure.parse
python -m nrl_microstructure.odds path/to/nrl.xlsx
python -m nrl_microstructure.history
python -m nrl_microstructure.figures
```
