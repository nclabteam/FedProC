# Losses

| Abbreviation | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | Name | 
| ------------ | - | - | - | - | - | - | - | - | - | - | -- | ----- | --- |
| MAE          | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 🗸 | None | Mean Absolute Error | 
| MSE          | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | None | Mean Squared Error |
| RMSE         | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | None | Root Mean Squared Error |
||||||||||||||
||||||||||||||
| MAPE         | 🗸 | ✗ | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | ✗ | OOS PS | Mean Absolute Percentage Error | 
| MdAPE        | 🗸 | 🗸 | 🗸 | ✗ | 🗸 | ✗ | ✗ | ✗ | ✗ | 🗸 | OOS PS |  Median Absolute Percentage Error|
| sMAPE        | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 🗸 | OOS PS | Symmetric Mean Absolute Percentage Error |
| sMdAPE       |||||||||||| Symmetric Median Absolute Percentage Error |
| msMAPE       | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | OOS PS | Modified Symmetric Mean Absolute Percentage Error |
| RMSPE        | 🗸 | ✗ | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | ✗ | OOS PS | Root Mean Squared Percentage Error |
| RMdSPE       |||||||||||| Root Median Square Percentage Error |
||||||||||||||
||||||||||||||
| RMSSE        | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 🗸 | ✗ | 🗸 | ✗ | IS PSs | Root Mean Squared Scaled Error |
||||||||||||||
||||||||||||||
| MALE         |||||||||||| Mean Absolute Log Error |
| EMALE        |||||||||||| Exponential Mean Absolute Log Error |
| RMSLE        |||||||||||| Root Mean Squared Log Error |
| ERMSLE       |||||||||||| Exponential Root Mean Squared Log Error |
||||||||||||||
||||||||||||||
| MQC          |||||||||||| Multi-Quantile Loss Change | 
| RSquared     |||||||||||| Coefficient of Determination | 
| sMAPC        |||||||||||| Symmetric Mean Absolute Percentage Change | 
| KLDivergence |||||||||||| Kullback–Leibler Divergence | 

**Note (columns mapping):**

1. Stationary Count Data (>>0)  
2. Seasonality  
3. Trend (Linear/Exp.)  
4. Unit Roots  
5. Heteroscedasticity  
6. Structural Breaks (With Scale Differences) in Forecast Horizon  
7. Structural Breaks (With Scale Differences) in Training Region  
8. Structural Breaks (With Scale Differences) in Forecast Origin  
9. Intermittence (zeros / intermittent series)  
10. Outliers (robustness)  
11. Scaling (scale-invariant)  
11.1. OOS: Out-of-Sample  
11.2. IS: In-Sample  
11.3. PS: Per Step  
11.4. PSs: Per Series  

Legend:
- 🗸 : generally suitable / recommended  
- ✗ : generally unsuitable or problematic without adjustments

## Metric scope

- `LOSSES` contains two-input objectives selectable for training.
- `EVAL_LOSSES` contains two-input scores, currently `RSquared`, that must not be minimized.
- `CONTEXT_LOSSES` contains `RMSSE`, `sMAPC`, and `MQC`; generic `evaluation_result()` excludes them.

`RMSSE` requires separate in-sample history for its seasonal-naive scale. `sMAPC` and `MQC` require forecasts for the same periods from consecutive forecast origins, not observations.

Generic evaluation also skips KL divergence and the positive-only log metrics; call those directly when their input-domain contract is satisfied.

References: [Forecast Evaluation for Data Scientists](https://arxiv.org/abs/2203.10716) · [On forecast stability](https://doi.org/10.1016/j.ijforecast.2025.01.006) · [On the stability of global forecasting models](https://arxiv.org/abs/2506.05776)
