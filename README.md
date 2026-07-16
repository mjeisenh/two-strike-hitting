# Shorten Up or Swing Away? A Machine-Learning Approach to Two-Strike Hitting

## Project Overview

Conventional baseball wisdom suggests that batters must fundamentally alter their swing mechanics when facing two strikes - commonly referred to as "shortening up," "protecting the plate," or just aiming to make contact. This project challenges that narrative by leveraging MLB Statcast bat-tracking data from the 2024 and 2025 regular seasons.

Using an unsupervised-to-supervised machine learning pipeline, this study:

* Identifies three distinct mechanical adjustment profiles using Principal Component Analysis (PCA) and K-Means Clustering.
  
* Predicts expected contact quality (xwOBA) using an XGBoost Regressor trained on pitch-level and swing-level attributes.
  
* Evaluates performance gaps across batter archetypes using post-hoc statistical testing.

### Project Findings

> Traditional coaching is costing teams runs. This modeling reveals that swing length is the least influential predictor of expected contact quality (6.2% feature importance), while bat speed is the single most critical driver of success (22.9% feature importance). Hitters who do not shorten up ("Free Swingers") outperform traditional "Committed Shorteners" by a statistically significant 19.7 points in predicted xwOBA.
> 
> 

---

## The Data Pipeline & Processing

The final modeling dataset was constructed from over 500,000 regular-season swings.

```
Raw Statcast Data (2024-2025) 
  └── Clean Non-Swings & Bunt Attempts 
        └── Apply Competitive Swing Filter (9.6% removed)
              └── Remove Switch Hitters (72,000 pitches)
                    └── Apply Min. Sample Threshold (40+ two-strike swings)
                          └── Final Dataset: 505,843 swings / 551 batters

```

* **Competitive Swing Filter**: Utilized Statcast's definition of a 'competitive swing', retaining the fastest 90% of each player's swings, and any swing over 60 mph resulting in an exit velocity over 90 mph. This mathematically removes check-swings, swords, and defensive noise.

* **Switch Hitter Exclusion**: Excluded 67 switch-hitters to preserve directionality interpretability (where negative values represent the pull side and positive values indicate the opposite field).

* **Target Sample Constraint**: Enforced a minimum threshold of 40 two-strike competitive swings per batter to suppress small-sample variance.

---

## Phase 1: Unsupervised Clustering (Identifying the Archetypes)

### Feature Engineering: The "Delta" Shift

To isolate how batters adjust rather than grouping them by baseline physical ability, five swing-tracking features were transformed into delta metrics:

$$\Delta_{\text{feature}} = \bar{X}_{\text{feature, strikes}=2} - \bar{X}_{\text{feature, strikes}<2}$$

These deltas measure changes in bat speed, swing length, attack angle, attack direction, and swing path tilt.

### PCA Decomposition

Because human swing mechanics are highly correlated (e.g., shortening a swing has a correlation of $r = 0.75$ with flattening the attack angle), PCA was applied to standardized features to eliminate multicollinearity. Three components were retained, capturing **92.2% of the total variance**:

* **PC1: Swing Aggression (57.0% variance)**: Captures swing length, attack angle, attack direction, and bat speed adjustments.


* **PC2: Independent Plane Adjustment (21.3% variance)**: Heavily dominated by swing path tilt.


* **PC3: Residual Speed (13.9% variance)**: Captures effort-level changes in bat speed independent of swing geometry.



### K-Means Results

An elbow plot and silhouette analysis identified $k = 3$ as the optimal clustering solution, revealing three distinct strategic approaches:

| Hitter Archetype | Batters ($n$) | Mechanical Profile Centroid (Two-Strike Delta Shift) | Notable Real-World Examples |
| --- | --- | --- | --- |
| **Free Swingers** | 160 | Minimal adjustments. Marginally increased swing length (+0.037 ft) and attack angle (+0.72°); minimal bat speed reduction (-0.92 mph); pulled the ball (-1.623°). | Giancarlo Stanton, Joey Gallo, Roman Anthony|
| **Moderate Adjustors** | 282 | Measured mechanical concessions. Slower bat speed (-1.387 mph) and shorter length (-0.093 ft).| Shohei Ohtani, Aaron Judge, Juan Soto, Yordan Álvarez |
| **Committed Shorteners** | 109 | Extreme defensive approach. Drastic speed reduction (-2.178 mph), shortened length (-0.241 ft), and opposite-field focus (+3.651°).| Steven Kwan, J.D. Martinez|
---

## Phase 2: Supervised Outcome Prediction (Quantifying Success)

To evaluate which archetype produces the most impactful outcome with two strikes, a regression model was built using pitch-level contact events.

### Target Variable: xwOBA (Expected Weighted On-Base Average)

Unlike box-score statistics (which penalize a batter for hitting a line drive directly into defensive positioning), xwOBA uses Statcast launch speed and launch angle to calculate the expected value of contact, isolating swing quality from defensive luck.

### Model Training & Validation

* **Inputs**: 5 bat-tracking metrics and 5 pitch-attribute control variables (speed, movement, and plate location).

* **Validation Scheme**: **5-Fold GroupKFold Cross-Validation** grouped by *batter*. This player-split approach ensures zero data leakage across folds, forcing the models to generalize to unseen player profiles.

* **Model Selection**: Three algorithms were compared, with **XGBoost** demonstrating superior predictive performance:

| Model | RMSE Mean | RMSE Std | MAE Mean | MAE Std |
| --- | --- | --- | --- | --- |
| Linear Regression | 0.3556 | 0.0049 | 0.2778 | 0.0035 |
| Random Forest | 0.3522 | 0.0048 | 0.2747 | 0.0031 |
| **XGBoost** | **0.3512** | **0.0049** | **0.2725** | **0.0034** |

### Feature Importance: Challenging Traditional Coaching

XGBoost feature importances highlight a stark mismatch between traditional strategy and objective data:

```
1. Bat Speed           █████████████████████ 22.9%
2. Plate X (Location)  ████████████ 13.3%
3. Attack Direction    ██████████ 10.7%
4. Attack Angle        ██████████ 10.6%
...
8. Swing Length        ██████ 6.2%

```

* **Bat Speed** is the single most crucial driver of high-quality contact (22.9%).

* **Swing Length**—the metric coaching traditionalists tell players to aggressively minimize—is the least important mechanical factor (6.2%).

---

## Archetype Performance Comparison

Evaluating the predicted two-strike xwOBA across the three clusters reveals a clear hierarchy:

```
Free Swinger        (Mean xwOBA: 0.3655)  ██████████████████████████████
Moderate Adjustor   (Mean xwOBA: 0.3566)  ████████████████████████████
Committed Shortener (Mean xwOBA: 0.3458)  ██████████████████████████

```

A **Kruskal-Wallis** test verified that these performance gaps are highly significant ($H = 19.22$, $p < 0.001$). Pairwise comparisons using the **Games-Howell** post-hoc test established:

* **Free Swinger vs. Committed Shortener ($p < 0.001$)**: Free Swingers produce a predicted xwOBA **19.7 points higher** than Committed Shorteners.

* **Free Swinger vs. Moderate Adjustor ($p = 0.041$)**: Free Swingers maintain a statistically superior profile for pure contact quality, even when compared against a cohort loaded with elite talent.

* **Moderate Adjustor vs. Committed Shortener ($p = 0.029$)**: Controlled, measured concessions heavily outperform the complete mechanical rebuild typical of Shorteners.

---

## Key Takeaways

1. **The "Contact Tax" is Real**: While shortening up might minimize the risk of a strikeout, it imposes a severe mechanical penalty on the bat speed and swing path required to hit the ball hard. The resulting balls in play are fundamentally degraded.

2. **Modern Pitching Demands Aggression**: In an era of elite high-velocity pitching and optimized defensive positioning, "protecting the plate" should not mean "abandoning the swing".

3. **Elite Hitters Find Balance**: Hitters like Shohei Ohtani and Aaron Judge do not abandon their identity to shorten their swings; they employ a measured, moderate adjustment that preserves core bat speed while slightly mitigating strikeout risk.
