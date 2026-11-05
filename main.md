# Abstract

*Respiratory-related health issues are increasingly becoming widespread
throughout the United States. Chronic Obstructive Pulmonary Disease
(COPD) is the fourth-leading cause of death in the U.S. and is projected
to rise in the ranks in the near future. There are a plethora of
important factors that, taken together, contribute to an explanation of
that startling fact. A full explanation, however, is incomplete without
an analysis concerning the extent to which different regions across the
United States are impacted. Therefore, in this paper, we seek to examine
specifically how environmental factors may help explain regional
disparities in respiratory-related mortality rates. Particularly, we
look to environmental variables like air pollution, loss of vegetation
density, and climate conditions, all of which are region-specific and
can help explain regional differences. Through an exploratory data
analysis, we examine geographic heat maps of the United States to
contextualize the correlative relationships between the aforementioned
environmental factors and respiratory-related crude mortality rates.
Further, we apply machine learning models, specifically Random Forest
and Gradient Boosting algorithms which capture non-linear relationships,
in order to determine which model best predicts our target variable:
crude mortality rate. We examine the performance metrics of each model
within each of the four Census regions: Northeast, Midwest, South, and
West. Ultimately, we learned that the Random Forest model outperforms
the Gradient Boosting model in each region, but the former is not
without its flaws. We also applied a risk classification model through a
Random Forest model, categorizing regions into high-risk and low-risk.
The model classified the Midwest and South as high-risk and the
Northeast and West as low-risk, which is entirely consistent with prior
research and literature as well as our own exploratory data analysis.
Ultimately, we acknowledge that our analysis in this report is not at
all determinative of any finding of a causal connection between
environmental factors and respiratory-related mortality. After all,
there are thousands of confounding variables that impact respiratory
health and mortality. Nevertheless, our analysis provides meaningful
insight into the predictive analysis of respiratory-related mortality as
well as the non-trivial relationships between regional crude mortality
rates and air pollution, vegetation density, and climate conditions.*

# Introduction

Today, it is crucial to view our atmosphere as a dynamic system, where
hundreds of distinct variables converge and interact. Operating within
this dynamic system will allow us to optimally analyze important
environmental factors that contribute to respiratory illness in the
United States. Chronic Obstructive Pulmonary Disease (COPD), rhinitis,
and respiratory infections are particularly pertinent respiratory
conditions throughout the US, with COPD being the fourth-leading cause
of death in the US and projected to be the third-leading cause in the
future [@berry2010mortality]. However, not all regions in the US are
treated equally. Different regions in the US may experience varying
health outcomes based on climatic conditions, vegetation coverage, and
air pollutant concentrations.

Air pollution remains a significant environmental and public health
concern in the United States, contributing to a wide range of adverse
health outcomes across different regions. Emissions from transportation,
industrial activity, power generation, and other sources release
pollutants such as fine particulate matter (PM2.5), nitrogen dioxide
(NO2), and ozone (O3) into the atmosphere [@meo2024environmental]. While
CO2 is primarily associated with climate change, it also serves as an
indicator of combustion-related activities that generate co-pollutants
known to affect respiratory health [@ALA_HealthImpactAirPollution].
Exposure to these pollutants has been linked to inflammation of the
respiratory system, reduced lung function, and increased susceptibility
to respiratory diseases like COPD [@duan2020air; @meo2024environmental].

PM2.5, NO2, and O3 are associated with both chronic and acute
respiratory conditions, with their health impacts shaped by
environmental context. Vegetation coverage can help mitigate exposure to
pollution through pollutant removal and temperature regulation
[@diener2021can; @IHME_PM25_2024; @HuangfuAtkinson2020], while climatic
factors such as temperature and seasonality influence pollutant
formation and respiratory risk. These factors vary widely across U.S
regions, contributing to geographic differences in pollution exposure
and respiratory health outcomes. The goal of this project is to examine
how air pollutants, vegetation coverage, and climatic conditions jointly
influence respiratory outcomes, including COPD, rhinitis, and
respiratory infections, across the United States.

Respiratory illness remains a major public health concern in the United
States, with conditions as COPD, rhinitis, and respiratory infections
contributing substantially to morbidity and mortality. Existing research
has demonstrated that exposure to air pollutants, including, PM2.5, NO2,
and O3, plays a significant role in worsening respiratory outcomes,
while climatic factors such as temperature extremes, wildfires, and
seasonal variability further amplify these
risks[@DAmato161-169; @doi:10.1152/physrev.00043.2022; @HuangfuAtkinson2020; @LEWY2024100313; @meo2024environmental].A
growing body of literature also suggests that environmental context,
particularly vegetation coverage, influences air quality, thermal
conditions, and population-level exposure to respiratory stressors.
However, much of the existing research has examined these factors in
isolation, despite evidence that air pollution, vegetation, and climate
interact within a dynamic atmospheric system. Large-scale, regionally
resolved analyses that jointly consider these environmental variables
across the United States remain
limited[@rs14051255; @su12031258; @DAmato161-169]. This study presents
an opportunity to address this gap by examining how air pollutants,
vegetation density, and climatic conditions together shape respiratory
health outcomes across regions, providing insight that may inform future
environmental health research and public health policy in a changing
climate.

## Central Research Question

How do air pollutants (PM2.5, NO2, and O3), vegetation density, and
climatic conditions interact across the United States to shape
respiratory health outcomes, specifically respiratory-related mortality?

### Sub-Question 1

Can machine learning models accurately predict regional respiratory
health outcomes using air pollutant levels, vegetation coverage, and
climatic variables?

### Sub-Question 2

Which environmental factors (PM2.5, NO2, O3, CO2 emissions, vegetation,
climate variables) are the most important predictors of respiratory
health outcomes in machine learning models?

### Sub-Question 3

Do machine learning models identify nonlinear relationships or exposure
thresholds between air pollutants and respiratory health outcomes that
linear models do not capture?

### Sub-Question 4

Can machine learning models classify U.S. regions into high-risk vs
low-risk respiratory health categories based on combined environmental
exposures?

# Methodology

## Data Processing and Integration

Our primary data [U.S. Environmental Protection Agency's Air Quality
System (AQS) monitoring database](https://www.epa.gov/aqs) which
provides detailed annual air quality and meteorological observations at
the county and state levels. We restricted our analysis to the years
2000 through 2016 in order to examine long-term environmental trends
while avoiding distortions associated with the COVID-19 period. The raw
AQS data were provided as separate annual files. These 17 yearly
datasets were first programmatically concatenated into a single unified
dataset to allow for consistent longitudinal analysis across time.

To achieve greater consistency across the years, we standardized the
column names and ensured that the data types were uniform. The
measurements of pollutants were kept in their reported units (such as
NO2 in parts per billion, O3 in parts per million, and PM2.5 in
micrograms per cubic meter). Where applicable, non-numeric entries and
placeholder symbols were converted into missing values and treated
appropriately. Given that air quality monitoring coverage is
inconsistent across all U.S. counties, we limited our state- and
county-level summaries to locations that had a sufficient number of
recorded observations during the study period. Missing values were not
imputed at this point; instead, we conducted aggregation procedures
using the available observations to avoid introducing any artificial
bias.

Following the data cleaning, we aggregated the information by state and
county to determine average pollutant concentrations and climate metrics
for the years 2000 to 2016. This aggregation allowed us to produce
comparative heat maps and regional summaries while retaining the ability
to analyze annual trends. Geographic identifiers were carefully
preserved to facilitate future merging with datasets concerning
vegetation and respiratory mortality.

Vegetation dataset was obtained from the [World Resources
Institute](https://docs.google.com/spreadsheets/d/1ahEdgtt3567sU8DJrEFMTzUqE47WLJ21/edit?gid=283952840#gid=283952840)
and provide county-level measures of annual tree cover loss. Because our
air quality analysis spans 2000--2016, the vegetation dataset was
truncated to the same time period to ensure temporal consistency. County
and state identifiers were standardized to match those in the AQS
dataset, allowing for accurate cross-dataset integration. It is
important to note that this dataset measures vegetation loss rather than
total vegetation density, which may influence interpretation of
environmental buffering effects.

Finally, the last dataset involves the respiratory-related crude
mortality rates by State and County from CDC Wonder. The crude mortality
rate measures the number of respiratory-related deaths per 1,000 people
in a given year. This is a crucial dataset for our purposes because we
want to analyze the crude mortality rate by region, side-by-side with
the regional concentrations of air pollutants. The CDC has provided a
comprehensive dataset of respiratory-related mortality rates by region
from 2000 to 2016, which is our target year range because we wanted to
be as detached as possible from the COVID-19 period.

This project combines environmental and health datasets at the county
level throughout the United States. The dataset includes metrics of air
pollution (NO2, PM2.5, and Ozone), climate factors (ambient and outdoor
temperature, wind speed and direction), vegetation indicators (tree
coverage loss), and respiratory health outcomes measured using crude
mortality rates.

Data processing involved cleaning column names, addressing missing
values, and merging datasets using county-level identifiers. Rows with
missing values in key variables were removed to ensure consistency
across analyses. The final dataset for modeling consisted of 1,122
counties.

## Introduction to Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to examine distributions,
identify outliers, and uncover regional trends. Visualizations such as
histograms and geographic maps revealed substantial variation in
environmental exposures and respiratory outcomes across the United
States.

In particular, Southern and Midwestern regions exhibited higher average
mortality rates compared to the Northeast and West, suggesting
geographic disparities in respiratory health risk. Additionally, the
distribution of mortality rates was right-skewed, indicating the
presence of extremely high-risk counties that may be more difficult to
predict.

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./distribution of crude mort.png" style="width:60.0%" />
<figcaption>Distribution of Crude Mortality Rates</figcaption>
</figure>

To better understand the behavior of the response variable, a
distribution plot of crude mortality rates was generated. The
distribution provides insight into the variability and spread of
mortality across counties, highlighting whether the data are skewed or
concentrated within a narrow range. Most of the county crude rates seem
to be concentrated between 80 and 120, while there are some notable
outlier counties near the 250 to 300 range. Understanding the shape of
the target variable is important for selecting appropriate modeling
techniques and interpreting prediction performance.

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 4.14.06 PM.png"
style="width:85.0%" />
<figcaption>Correlation Heatmap</figcaption>
</figure>

A correlation heatmap was used to examine relationships between
environmental variables and mortality. Interestingly, nitrogen
dioxide(NO2) shows a moderate negative correlation with crude mortality
rates (-0.39), suggesting that counties with higher NO2 levels tend to
have lower mortality. While this is counterintuitive given the known
health impacts of air pollution, it likely reflects underlying
confounding factors such as urbanization, healthcare access, and
socioeconomic differences.

PM2.5 shows little to no correlation with mortality, indicating that its
effects may not be captured through simple linear relationships or may
be influenced by other interacting variables. Climate variables,
including temperature, display weak negative relationships with
mortality, while vegetation loss shows a slight positive association.
Overall, these findings highlight the complexity of environmental health
relationships and reinforce the need for multivariate modeling
approaches.

## Introduction to Correlative Analysis and Machine Learning Models

To examine the relationships between environmental variables and
respiratory health outcomes, a correlation analysis was carried out. The
results revealed that pollution variables, including NO2 and PM2.5, are
associated with mortality, although these associations are not strongly
linear. Climate and vegetation variables exhibited more complex
interactions, emphasizing the importance of multivariate modeling.

To capture nonlinear relationships, machine learning models were
implemented. Random Forest and Gradient Boosting were selected due to
their ability to model complex interactions between variables. The
dataset was split into training (70%), validation (15%), and testing
(15%) sets. Model performance was evaluated using metrics including
accuracy, balanced accuracy, ROC-AUC, and F-1 score. Random Forest was
expected to outperform other models due to its robustness and ability to
handle nonlinear relationships and feature interactions.

## Introduction to Risk Classification

To identify high-risk regions, counties were classified into binary
categories based on the 75th percentile of crude mortality rates.
Counties at or above this threshold were labeled as high risk, while
those below were categorized as low risk. This resulted in an imbalanced
dataset, with a larger proportion of counties classified as high risk. A
Random Forest classifier with SMOTE was implemented to address class
imbalance, and the optimal classification threshold was selected based
on maximizing the F1-score for the high-risk class. The final model
achieved an AUC of 0.908, indicating strong overall discriminatory
performance between high and low-risk counties.

# EDA Results

## Air Pollution

### NO2 Concentrations by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.28.34 AM.png"
style="width:105.0%" />
<figcaption>Average NO2 Concentrations by State (parts per billion) from
2000-2016</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.26.41 AM (1).png"
style="width:100.0%" />
<figcaption>Average NO2 Concentrations by County (parts per billion)
from 2000-2016</figcaption>
</figure>

It is immediately apparent that there is a stark contrast in NO2
concentrations between several states. For example, in Montana, Wyoming,
North Dakota, and South Dakota, we see the lowest levels of NO2 in the
nation. On the other hand, in Arizona, Illinois, West Virginia, New
Jersey, and New York, we see the highest concentrations of NO2 in the
nation. While further research would be helpful in determining exactly
why these drastic differences exist, the preliminary research and
general knowledge at our disposal tells us that these observations are
not at all unusual. Montana, Wyoming, North Dakota, and South Dakota
indeed have the lowest population densities of any other state among the
contiguous 48 states (Alaska is the absolute lowest) [@Statista].
However, it is worth noting that Alaska has relatively high levels of
NO2. This likely because NO2 concentrations are usually the highest in
cooler climates because we typically see increased energy usage for
heating during the cooler seasons. Cooler climates also create
temperature inversions, trapping NO2 near the ground, preventing them
from dispersing [@Brown_2025]. Furthermore, the aforementioned high-NO2
states are also very densely populated, but that fact alone would not be
fully sufficient to explain the high concentrations of NO2 in those
regions because Florida and Massachusetts are also very densely
populated. There are likely other factors like traffic congestion and
climate that might contribute to a fuller explanation of this
phenomenon.

The county map doesn't seem nearly as helpful as the state map, but it
nevertheless provides important insight into how smaller regions differ
in NO2 concentrations. It is immediately obvious that large counties
like Maricopa County in Arizona and San Bernardino and Los Angeles in
California have particularly high levels of NO2 concentration. This is
certainly to be expected since those counties are indeed full of traffic
congestion and dense populations. For the most part, an analysis of the
available counties is rather consistent with what we would expect. While
each individual county may not have an unusually high level of NO2, in
the aggregate groups of counties can collectively have high
concentrations that contribute to their respective state's overall air
quality.

### PM2.5 Concentrations by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.33.13 AM (1).png"
style="width:120.0%" />
<figcaption>Average PM2.5 Concentrations by State (micrograms/cubic
meter) from 2000-2016</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.35.16 AM (1).png"
style="width:100.0%" />
<figcaption>Average PM2.5 Concentrations by County (micrograms/cubic
meter) from 2000-2016</figcaption>
</figure>

Looking at the State map, we already see a vastly different heat map
than the NO2 map. The most highly PM2.5-concentrated states include West
Virginia, Indiana, Kentucky, Tennessee, Alabama, and Georgia, while the
least concentrated states include Wyoming and North Dakota. In the
former region, those states are marked by a combination of heavy
industrial activity, high traffic density, power plants, and other
PM2.5-related emissions. California is an interesting case because as a
whole it does not seem to be particularly high in PM2.5 concentration,
but when looking at its counties, it is easily apparent that only a
handful of densely populated counties account for the highest levels of
PM2.5. In Northern California, we see much more mild levels of PM2.5
which balance out the much higher levels we see in the dense south.
Comparatively, now looking at the Midwest, South, and Mid-Atlantic
regions, we see a much more substantial number of counties that are each
relatively high in PM2.5 concentration. Whereas NO2 levels in each
county in this region are relatively low, the PM2.5 levels in each
county in the same region are relatively high, contributing to the
overall PM2.5 concentrations of the states in which they reside.

### O3 Concentrations by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.38.08 AM (1).png"
style="width:120.0%" />
<figcaption>Average Ozone Concentrations by State (parts per million)
from 2000-2016</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.39.20 AM (1).png"
style="width:100.0%" />
<figcaption>Average Ozone Concentrations by County (parts per million)
from 2000-2016</figcaption>
</figure>

Looking at the State map, it is clear that the states with the highest
ozone concentrations include Arizona, Utah, Colorado, Idaho, and
Wyoming. We see a similar trend when looking at the county map, where a
lot of the counties in the west seem to have relatively high ozone
concentrations. These state and county trends are largely consistent
with our prior research. In our literature review, we learned that
exposure to O3 is usually highest on hot summer days
[@beckett1991ozone]. Certainly, the aforementioned states are drier and
have hotter average temperatures than most other states. Notably,
Alaska, which seems to have the lowest level of ozone concentration in
the United States, has a particularly cool climate, which accounts for
it also having a particularly low ozone concentration. It is also worth
noting that ozone levels are relatively high in the South and
Mid-Atlantic regions, where PM2.5 levels are also incredibly high. Given
these facts, we can safely conclude that those regions seems to most
prone to poorer air quality than most other states, given that ozone and
PM2.5 are both potent and prevalent air pollutants not only in the U.S.
but around the world.

## Vegetation Density

### Vegetation Density Loss in 2001 by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-07 at 9.16.57 AM.png"
style="width:80.0%" />
<figcaption>The Loss of Tree Coverage (in hectares) by State in
2001</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.45.28 AM.png"
style="width:100.0%" />
<figcaption>The Loss of Tree Coverage (in hectares) by County in
2001</figcaption>
</figure>

The 2001 maps illustrates that the distribution of tree cover loss was
not uniform across the country. Areas with higher levels of vegetation
loss are notably found in the Pacific Northwest, Northern California,
certain regions of the Southeast, and Alaska. In contrast, many counties
in the Midwest and Northeast display relatively lower levels of tree
cover loss during that year.

The pattern of vegetation loss in heavily forested areas is consistent
with the dynamics of logging activities, wildfire occurrences, and
shifts in land use. In the Southeast, where forestry is economically
vital, the heightened vegetation loss likely mirrors the cycles of
timber harvesting. In the western states, the decrease in vegetation may
be attributed to wildfire activity and environmental stress linked to
drought. These regional distinctions are crucial for interpreting
pollutant concentrations, as vegetation functions as a natural buffer
that can diminish particulate matter buildup and enhance local air
quality.

### Vegetation Density Loss in 2016 by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-07 at 9.17.05 AM.png"
style="width:80.0%" />
<figcaption>The Loss of Tree Coverage (in hectares) by State in
2016</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.48.45 AM.png"
style="width:100.0%" />
<figcaption>The Loss of Tree Coverage (in hectares) by County in
2016</figcaption>
</figure>

By 2016, vegetation loss appears more pronounced and geographically
widespread in certain western and southeastern regions compared to 2001.
Parts of California, the Pacific Northwest, and Alaska show intensified
levels of tree cover reduction. Several southeastern counties also
continue to demonstrate persistent vegetation loss.

The increased vegetation loss observed in western states may be
partially attributable to heightened wildfire activity and prolonged
drought conditions during the mid-2010s. Meanwhile, sustained vegetation
reduction in the Southeast may reflect continued forestry and land
development practices. While vegetation loss does not directly measure
air pollution levels, reductions in tree coverage may diminish the
environment's capacity to absorb pollutants such as PM2.5 and ozone
precursors. This reduction in natural filtration could contribute to
greater pollutant persistence in affected regions.

## Climate Conditions

### Temperature Trends by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Average Temp by state (1).png" style="width:100.0%" />
<figcaption>Average Ambient Temperature by State
(2000–2016)</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Average temp by county (1).png" style="width:100.0%" />
<figcaption>Average Ambient Temperature by County
(2000–2016)</figcaption>
</figure>

It is immediately apparent that southwestern states such as Arizona,
Nevada, Texas, and parts of California exhibit the highest average
ambient temperatures over the study period. In contrast, northern states
such as Montana, North Dakota, Minnesota, and Maine display
substantially lower annual averages. This geographic gradient is
consistent with established climatic zones across the United States.

At the county level, variation becomes more pronounced. Desert counties
in Arizona and Nevada consistently show some of the highest average
temperatures in the nation, while mountainous and northern counties
exhibit significantly lower averages. These findings are particularly
important when considered alongside ozone concentrations, as higher
temperatures promote ozone formation through enhanced photochemical
reactions. The spatial overlap between high-temperature regions and
elevated ozone concentrations supports this well-documented atmospheric
relationship.

### Wind Speed by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./wind speed by state.png" style="width:100.0%" />
<figcaption>Average Wind Speed by State (2000–2016)</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./wind speed by county.png" style="width:100.0%" />
<figcaption>Average Wind Speed by County (2000–2016)</figcaption>
</figure>

The data indicate that Great Plains and Midwestern states generally
exhibit higher average wind speeds compared to coastal urban regions and
geographically enclosed western counties. This pattern may help explain
why certain industrial or agricultural states do not consistently
exhibit the highest pollutant concentrations despite ongoing emissions
activity.

In contrast, metropolitan regions characterized by lower wind
circulation patterns may experience pollutant buildup, particularly for
NO2 and PM2.5. Reduced atmospheric mixing increases the likelihood of
prolonged exposure in densely populated areas, thereby increasing
potential health risks.

### Atmospheric Pressure by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./pressure by state (1).png" style="width:100.0%" />
<figcaption>Average Atmospheric Pressure by State
(2000–2016)</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./pressure by county (1).png" style="width:100.0%" />
<figcaption>Average Atmospheric Pressure by County
(2000–2016)</figcaption>
</figure>

While atmospheric pressure does not vary as dramatically geographically
as temperature, certain western and northern states exhibit seasonal
conditions consistent with inversion-prone environments. These
atmospheric stability conditions may help explain elevated NO2 and PM2.5
concentrations in some lower-population regions, as trapped air masses
limit pollutant dispersion.

## Respiratory-Related Crude Mortality Rates by State and County

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.42.21 AM.png"
style="width:105.0%" />
<figcaption>Average Respiratory-Related Crude Mortality Rate by State
from 2000-2016</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-03-06 at 11.43.22 AM.png"
style="width:100.0%" />
<figcaption>Average Respiratory-Related Crude Mortality Rate by County
from 2000-2016</figcaption>
</figure>

In Figures 19 and 20, respiratory-related crude mortality rates seem to
be the highest in the Midwest and Southeast regions of the United
States. For the most part, this tracks relatively well with our previous
observations about vegetation loss, climate conditions, and especially
air pollution. The crude rates are especially high West Virginia where
there are somewhat high ozone concentrations and very high PM2.5 and NO2
concentrations. The South and parts of the Midwest also have higher
levels of vegetation loss compared with most of the country. It is also
worth noting that Alaska has the lowest respiratory-related crude rates.
This makes sense considering that air pollutant concentrations in Alaska
are not nearly as high as many of the Southern and Midwest countries,
despite it having relatively high levels of vegetation loss. Moreover,
average atmospheric pressure and ambient temperature are particularly
higher in South and Midwest than other regions, which gives us insight
into how these climate conditions may be correlated with crude mortality
rates in those regions.

# Machine Learning Results: Random Forest vs. Gradient Boosting Algorithms

## Regional Analysis

### Northeast

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 1.34.53 PM.png"
style="width:95.0%" />
<figcaption>Northeast Model Comparison</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 2.20.54 PM.png"
style="width:110.0%" />
<figcaption>Northeast Feature Importance Comparison</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 5.26.42 PM.png"
style="width:80.0%" />
<figcaption>Northeast Performance Metrics Comparison</figcaption>
</figure>

When looking at the Northeast, Figures 21, 22, and 23 indicate that
Random Forest performed better. In Figure 21, the $R^2$ value is more
significant than the gradient boosting, and comparing that to the other
metrics shown in Figure 23, it is safe to assume that Random Forest
performed better. Moreover, in Figure 22, both models prioritized
different features. The Gradient Boosting model's most important feature
was tree cover loss and the Random Forest Model's most important feature
was ozone concentration.

### Midwest

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 1.36.33 PM.png"
style="width:100.0%" />
<figcaption>Midwest Model Comparison</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 2.24.24 PM.png"
style="width:115.0%" />
<figcaption>Midwest Feature Importance Comparison</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 5.27.33 PM.png"
style="width:80.0%" />
<figcaption>Midwest Performance Metrics Comparison</figcaption>
</figure>

When looking at the Midwest region, Figures 24, 25, and 26 indicate that
Random Forest performed better. In Figure 24, the $R^2$ value is more
significant than the gradient boosting, and comparing that to the other
metrics shown in Figure 25 (especially MAE and MAPE), it is safe to
assume that Random Forest performed better. Moreover, in Figure 26, both
models prioritized the same feature: NO2.

### South

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 1.38.44 PM.png"
style="width:100.0%" />
<figcaption>South Model Comparison</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 2.26.52 PM.png"
style="width:110.0%" />
<figcaption>South Feature Importance Comparison</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 5.28.30 PM.png"
style="width:80.0%" />
<figcaption>South Performance Metrics Comparison</figcaption>
</figure>

When looking at the South, Figures 27, 28, and 29 indicate that Random
Forest performed better. In Figure 27, the $R^2$ value is more
significant than the gradient boosting, and comparing that to the other
metrics shown in Figure 29, it is safe to assume that Random Forest
performed better. Shifting focus to Figure 28, both models highlight NO2
as a key predictor, suggesting that traffic-related emissions and
atmospheric conditions play a significant role in shaping respiratory
risk in this region.

### West

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 1.40.24 PM.png"
style="width:110.0%" />
<figcaption>West Model Comparison</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 2.29.15 PM.png"
style="width:110.0%" />
<figcaption>West Feature Importance Comparison</figcaption>
</figure>

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 5.29.12 PM.png"
style="width:80.0%" />
<figcaption>West Performance Metrics Comparison</figcaption>
</figure>

Finally, looking at the Western region, Figures 30, 31, and 32 also
indicate that Random Forest performed better. In Figures 30 and 32, the
$R^2$ values and the other metrics perform better in the Random Forest
model than the Gradient Boosting model. Looking at Figure 31, both
models highlight wind speed as a key predictor, indicating that
atmospheric conditions play a significant role in shaping respiratory
risk in this region

## Overall: The United States

### Gradient Boosting Algorithm and Feature Importance

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 1.45.04 PM.png"
style="width:100.0%" />
<figcaption>Gradient Boosting Algorithm and Feature Importance as
Applied to the United States</figcaption>
</figure>

Figure 33 presents the comparison between actual and predicted mortality
values through the Gradient Boosting algorithm. We see a concentrated
clustering of points in the lower ranges of mortality rates with a wider
spread of points in the higher ranges. It is clear that the model is
more sensitive to noise and variability, especially when compared to the
Random Forest model below. Moreover, the most influential predictor in
the Gradient Boosting algorithm, as applied to the whole United States,
was NO2.

### Random Forest and Feature Importance

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Screenshot 2026-04-25 at 1.42.45 PM.png"
style="width:100.0%" />
<figcaption>Random Forest Model and Feature Importance as Applied to the
United States</figcaption>
</figure>

Figure 34 presents the comparison between actual and predicted mortality
values. The clustering of points around the diagonal indicates that the
model captures overall trends, though noticeable dispersion reflects
prediction error for certain counties. Further, feature importance
analysis reveals that nitrogen dioxide (NO2) is the most influential
predictor, followed by temperature-related variables, ozone, wind speed,
and PM2.5. The prominence of NO2 is consistent with the correlation
analysis, where it exhibited a strong (though negative) relationship
with mortality. This suggests that NO2 may be capturing broader
structural factors such as urbanization rather than acting solely as a
direct pollutant indicator.

# Risk Classification Results

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./risk classifciation confusion.png" style="width:80.0%" />
<figcaption>Risk Classification Confusion Matrix</figcaption>
</figure>

The confusion matrix (Figure 31) shows that the model correctly
classified the majority of counties, achieving an overall accuracy of
0.92. Specifically, 148 low-risk counties and 8 high-risk counties were
correctly justified, while 10 low-risk counties were misclassified as
high-risk and 3 high-risk counties were misclassified as low risk.

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./Risk ROC.png" style="width:80.0%" />
<figcaption>Risk Classification ROC Curve</figcaption>
</figure>

The ROC curve (Figure 32) further demonstrates strong model performance
with a curve that closely approaches the top-left corner, reflecting a
favorable balance between true and false positive rates.

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./RISK feature importance.png" style="width:80.0%" />
<figcaption>Risk Classification Feature Importance</figcaption>
</figure>

Feature importance analysis (Figure 33) indicates that nitrogen dioxide
is the most influential predictor, followed by tree coverage loss,
PM2.5, and temperature-related variables.

<figure id="fig:placeholder" data-latex-placement="H">
<img src="./risk comparison.png" style="width:80.0%" />
<figcaption>Risk Classification Regional Distributions</figcaption>
</figure>

The regional analysis (Figure 34) shows variation in predicted risk
across U.S regions. The South and Midwest exhibit higher proportions of
predicted high-risk counties compared to the West and Northeast.

# Discussion

## EDA

As we previously surveyed different environmental factors across the
United States, we will now evaluate those results against
respiratory-related crude mortality rates to support our subsequent
findings from our machine learning and risk classification models.

First, we will discuss air pollution. According to the heat map of NO2
concentrations in the United States we discussed earlier, we see the
highest concentrations of NO2 in parts of the Midwest and most of the
Northeast. Some counties in Arizona and California, which are densely
populated, also show high levels of NO2. Moreover, the heat map of PM2.5
concentrations show the highest levels in the South and Midwest regions
as well as a few densely populated counties in California. Finally, O3
concentrations are highest in the western states, like Utah, Arizona,
Colorado, and parts of California. It is also worth noting that the
ozone concentrations in the South and Midwest regions are not at all
insubstantial. Taking all our observations about air pollution together,
we arrive at a clearer understanding of the crude mortality rate heat
map. We see the highest mortality rates in the Midwest and parts of the
South, specifically West Virginia, where PM2.5 and NO2 concentrations
are notably high. It is also worth addressing the relatively low crude
mortality rates near the West Coast, namely California and Arizona. The
air pollutant concentrations in those States and the counties therein
are noticeably high, yet the crude mortality rates are relatively low.
This apparent discrepancy might be explained by the lack of sufficient
county data in those regions or, more likely, the effects of confounding
variables like climate and vegetation density. Nevertheless, from this
observational analysis of the heat maps, we derived meaningful insight
into the regional relationships between air pollution and
respiratory-related crude mortality rates in the United States.

We will now move on to a discussion about vegetation density in the
United States to help better explain the mortality trends we observed.
In 2001 and 2016, the loss of tree coverage was highest in the South and
Northwest regions, with an interesting outlier observation in Maine and
Alaska. Most notably, however, was the substantial increase in
vegetation loss in California, the Northwest, and Alaska since 2001, as
observed in the 2016 heat map. We hypothesized that the loss of tree
coverage would support an unhealthy environment prone to
respiratory-related health issues, which in turn could lead to death.
Our data seems to support our hypothesis, given the relatively high
crude mortality rates in parts of northern California, the South, and
Maine. Our tree-coverage-based observations are, by no means, a perfect
reflection of respiratory-related mortality, and we certainly would not
expect it to be perfect. We are merely recognizing the presence of a
non-trivial correlative relationship between vegetation loss and
respiratory-related mortality rate. Notably, most of the United States
has not suffered from a substantial loss of tree-coverage, the regions
of which overlap with where we observed the highest crude mortality
rates. Thus, just because there was no substantial vegetation loss in a
particular region, it does not follow that that region would not be at
high risk of respiratory-related mortality. There are always more
variables to consider.

Finally, we will survey our climate conditions factors as it relates to
respiratory-related crude mortality rates to complete our preliminary
analysis and arrive at a more comprehensive conclusion about our overall
observations. From the ambient temperature heat map, it is clear that
from coast to coast, the southern half of the United States has higher
temperatures than the northern half, which of course is consistent with
general temperature trends in the United States. We do not, however, see
such segmented regional trends in wind speed by state and county, but
there are still some important observations to note despite the
substantial lack of county data. We see the highest relative wind speeds
in the Midwest region and, notably, West Virginia, whereas the lowest
wind speeds reside in the South and parts of the Northeast. Finally, the
atmospheric pressure heat map reveals a sharp regional contrast. The
eastern part of the United States has the highest atmospheric pressure,
while many states in the West, excluding the coastal states (California,
Oregon, and Washington), have relatively low levels of atmospheric
pressure. Taking all of these climatic observations together, when
comparing them side by side with respiratory-related crude mortality
rates, we see slightly similar geographic trends. The eastern part of
the United States, particularly the South and Midwest where atmospheric
pressure is high, wind speeds are low, and temperatures are relatively
warmer, is replete with higher mortality rates. Concurrently, the
non-coastal regions of the West, where atmospheric pressure is low and
ambient temperatures are high, we see lower crude mortality rates. These
climatic trends, of course, are not determinative on respiratory-related
mortality, but they certainly capture the overall mortality trends we
observed, especially when taken together with air pollution and
vegetation density variables.

From what can be gleaned from our exploratory data analysis above, we
can see how different environmental factors may help contribute to an
explanation about the general respiratory-related mortality trends we
see across the United States. Air pollution concentrations are strongest
in the South and Midwest, with a few outlier regions in California and
New England. Vegetation density loss is most prevalent in the South and
Northwest regions of the United States, where we would in turn expect
higher mortality rates. Moreover, the various climate variables we
observed interestingly coincided with the general regional mortality
trends, but our observations were by no means a perfect reflection of
those trends. Ultimately, our analysis here importantly contextualizes
the bulk of our further regional analysis through machine learning and
risk classification models. Next, we will discuss those models in detail
and reflect on their implications.

## Machine Learning Models and Performance Comparison

To evaluate predictive performance, Gradient Boosting and Random Forest
models were applied across four U.S regions: Northeast, Midwest, South,
and West. The performance of the model was assessed using $R^2$,
adjusted $R^2$, Mean Absolute Error(MAE), mean absolute probability
Error(MAPE), and Median Absolute Error(MedAE). Across all regions, the
Random Forest model consistently outperformed Gradient Boosting in terms
of predictive accuracy and error reduction

In the Northeast region of the United States, both models reveal
relatively strong predictive capabilities; however, Random Forest
consistently outperforms Gradient Boosting across the majority of the
metrics. Random Forest achieves a higher test $R^2$ of 0.6647, in
contrast to 0.5782 for Gradient Boosting, indicating that it accounts
for a greater proportion of variance in mortality outcomes on unseen
data. The higher training $R^2$ values (0.8089 for Random Forest and
0.7053 for Gradient Boosting) suggest that both models fit the training
data well, yet the difference between training and test performance
indicates a degree of overfitting, particularly for Random Forest. Error
metrics further validate Random Forest's advantage, with lower MAE
(9.1458 compared to 10.9276), MAPE(9.15% versus 10.93 %), and RMSE
(12.81 against 14.36), indicating more accurate and consistent
predictions overall, although Gradient Boosting achieves a slightly
lower median absolute error (5.51 versus 6.95), suggesting fewer extreme
individual errors. The feature importance results indicate that ozone
and tree cover loss are the primary predictors in this area, with
nitrogen dioxide (NO2) also contributing significantly. The prominence
of ozone suggests that photochemical pollution plays a crucial role in
respiratory outcomes in the Northeast, while the influence of vegetation
loss underscores the impact of environmental degradation. Random
Forest's ability to incorporate multiple contributing variables likely
explains its superior performance compared to Gradient Boosting, which
places greater emphasis on a limited number of dominant features.

In the Midwest, the Random Forest model clearly outperforms the Gradient
Boosting model in every performance metric. In our preliminary ML
analysis, we learned that the Random Forest model had a substantially
larger train $R^2$ value (0.7411) than the Gradient Boosting model
(0.1012). Moreover, the Random Forest test $R^2$ value (0.523), while
lower than its train value, was significantly larger than the Gradient
Boosting test $R^2$ value (0.1154). The other metrics additionally
supported the finding that the Random Forest model was superior. The
mean absolute error and mean absolute percentage error were both lower
in the Random Forest model (12.2391 compared to 15.1303 for Gradient
Boosting). Moreover, the root mean squared error (RMSE) was lower for
the Random Forest Model (14.8257 compared to 20.1889). Taking all of
these metrics together, it is abundantly clear that the Random Forest
model predicts respiratory-related crude mortality rates in the US
better than the Gradient Boosting model, exhibiting much lower error
rates. It is also worth noting that nitrogen dioxide contributed most
significantly to both models' predictive analysis. The primary
difference, though, is the larger NO2 proportion in the Gradient
Boosting model, whereas the Random Forest model more heavily
incorporated other features like PM2.5 and tree cover loss. Again, the
Random Forest model's ability to incorporate multiple contributing
variables may explain its superior performance compared to Gradient
Boosting, which places greater emphasis on a limited number of dominant
features.

In the Southern region, both models show predictive effectiveness, with
Random Forest once again exceeding Gradient Boosting. Random Forest
achieves a higher test $R^2$ of 0.5109 compared to 0.4060, which
indicates a stronger capacity to account for variability in mortality
outcomes. The training $R^2$ values (0.7952 for RF and 0.6203 for GB)
suggest that both models fit the training data reasonably well; however,
the differences between training and test performance indicate some
overfitting. Error metrics consistently favor Random Forest, with a
lower MAE (13.0014 versus 14.7951), MAPE (13% against 14.8%), and RMSE
(17.85 compared to 19.67), reflecting improved prediction accuracy and
reduced error magnitude. Additionally, Random Forest achieves a notably
lower median absolute error (7.87 compared to 11.66), indicating greater
strength against outliers. Feature importance analysis shows that both
nitrogen dioxide (NO2) and wind speed are significant predictors in the
South, with temperature and vegetation-related variables also
contributing. The importance of wind-related variables suggests that
atmospheric transport and pollutant dispersion are critical factors in
determining respiratory risk in this region. Random Forest's ability to
incorporate these multiple interacting environmental factors likely
enhances its performance, while Gradient Boosting's more concentrated
feature weighting may limit its capacity to fully capture these
dynamics.

In the West, the Random Forest model outperforms the Gradient Boosting
model in every performance metric, but the differences are less
substantial than, say, the Midwest models. In our preliminary ML
analysis, we learned that the Random Forest model had a larger train
$R^2$ value (0.6878) than the Gradient Boosting model (0.4469).
Moreover, the Random Forest test $R^2$ value (0.4772), while lower than
its train value, was slightly larger than the Gradient Boosting test
$R^2$ value (0.4125). The other metrics additionally supported the
finding that the Random Forest model was superior. The mean absolute
error and mean absolute percentage error were both lower in the Random
Forest model (11.1107 compared to 12.1393 for Gradient Boosting), but
the difference here was relatively insubstantial. Further, the median
absolute error for the Random Forest model was lower (7.2932 compared to
9.2382). Moreover, the root mean squared error (RMSE) was lower for the
Random Forest Model (15.534 compared to 16.4684). Taking all of these
metrics together, it is abundantly clear that the Random Forest model
predicts respiratory-related crude mortality rates in the US better than
the Gradient Boosting model, exhibiting much lower error rates. It is
also worth noting that wind speed (one of our climate variables)
contributed most significantly to both models' predictive analysis. The
Random Forest model, however, more meaningfully incorporates other
important features like PM2.5, Ozone, and NO2 (all air pollutants). Once
again, the Random Forest model's ability to incorporate multiple
contributing variables might explain its superior performance compared
to Gradient Boosting, which places greater emphasis on a limited number
of dominant features.

## Risk Classification

The results of the risk classification model demonstrate strong
predictive performance and reveal meaningful patterns in respiratory
health risk across the United States.

The model demonstrated a high AUC of 0.908, which reflects a strong
capacity to differentiate between high- and low-risk counties across a
spectrum of classification thresholds. Additionally, the overall
accuracy of 0.92 implies that the model is effective in accurately
classifying counties in general. However, given the dataset's imbalanced
characteristics, accuracy alone is insufficient for evaluating the
model's performance.

More significantly, the model achieved a recall of 0.73 for the
high-risk class, indicating that roughly 73% of high-risk counties were
correctly identified. This outcome is crucial, as overlooking high-risk
counties (false negatives) would diminish the model's relevance in a
public health framework. The confusion matrix further substantiates
this, indicating that only 3 high-risk counties were misclassified as
low risk.

On the other hand, precision for the high-risk class was lower at 0.44,
suggesting that a segment of counties predicted as high risk were
actually low risk. This tradeoff between precision and recall is
expected, particularly in light of the use of SMOTE and the priority
given to identifying high-risk regions. The corresponding F1-score of
0.55 illustrates this balance, capturing both the model's capability to
detect high-risk counties and the incidence of false positives.

Geographically, the model identifies the South and Midwest as regions
with the highest proportions of high-risk counties, with approximately
11% and 10% of counties deemed high risk, respectively. In contrast, the
Northeast has the lowest proportion (around 4%). These findings are in
line with established regional disparities in environmental conditions
and health outcomes, where southern regions often face higher pollution
levels and increased susceptibility to respiratory diseases.

Results regarding feature importance further highlight the significance
of environmental factors. Nitrogen dioxide (NO2) emerged as the most
critical predictor, with an importance score of approximately 0.31,
significantly surpassing all other variables. This indicates that
traffic-related air pollution is a major factor influencing respiratory
risk. Additional contributors, such as PM2.5 and the loss of tree cover,
further underscore the combined effects of air quality and environmental
degradation on health outcomes.

# Conclusions

This analysis examined how air pollution, vegetation loss, and climatic
conditions interact to influence respiratory-related mortality across
the United States. By integrating environmental and health data sets and
applying both exploratory analysis and machine learning models, this
research provides a comprehensive, regionally grounded understanding of
respiratory health risk.

In response to the central question, the results indicate that
respiratory health outcomes are influenced by the interaction of
multiple environmental factors rather than any single variable. Air
pollutants, particularly nitrogen dioxide (NO2), PM2.5, and ozone,
consistently emerged as key contributors, while the loss of vegetation
and climate variables such as temperature and wind further shaped
regional exposure patterns. These findings reinforce the significance of
viewing environmental health within a dynamic system where pollution,
climate, and land conditions interact to affect respiratory outcomes.

To address the sub-question, the results demonstrate that machine
learning models can effectively predict respiratory mortality using
environmental variables, with Random Forest consistently outperforming
Gradient Boosting across all regions in terms of $R^2$ and error
metrics, indicating a stronger predictive capability and generalization.
Feature importance analysis further reveals that nitrogen dioxide (NO2)
is the most influential predictor across both regression and
classification models, with additional contributions from PM2.5, ozone,
vegetation loss, and climate variables such as temperature and wind,
highlighting the combined role of air quality and environmental context.
The improved performance of machine learning models compared to simple
correlation analysis also suggests the presence of nonlinear
relationships and interactions between variables, as patterns that
appeared weak or counterintuitive in the exploratory analysis were more
effectively captured in multivariate models. Finally, the risk
classification model successfully identified high- and low-risk
counties, achieving strong performance (AUC = 0.908, accuracy = 0.92)
and revealing clear regional disparities, with the South and Midwest
exhibiting higher proportions of high-risk counties compared to the
Northeast and West.

Despite these strengths, it is important to recognize several
limitations. Firstly, the vegetation dataset focuses on tree cover loss
instead of total vegetation density, which may not adequately reflect
the protective benefits of green spaces. Secondly, the analysis is
confined to counties with adequate data availability, which may
inadvertently exclude rural or less-monitored areas, leading to
potential sample bias. Furthermore, the models do not take into account
socioeconomic status, healthcare access, or demographic variables, all
of which are known to affect respiratory health outcomes. Lastly,
although machine learning models demonstrate robust predictive
capabilities, they do not establish causal links between environmental
exposures and mortality.

Future research could build upon this analysis by integrating additional
variables such as socioeconomic factors, healthcare systems, and
demographic information to provide a more nuanced understanding of
environmental risks. Incorporating more recent data and
higher-resolution spatial data could enhance the accuracy and
applicability of the models. Moreover, employing causal inference
techniques or time-series methodologies may assist in unraveling the
intricate relationships between environmental exposures and respiratory
health outcomes. Increasing access to more detailed health datasets,
including hospitalization records or disease-specific information, would
also facilitate a more thorough evaluation of respiratory health beyond
mere mortality statistics.

Ultimately, this research confirms that environmental conditions are
essential in influencing respiratory health outcomes in the United
States. By merging air pollution, vegetation, and climate factors within
a machine learning framework, the study not only enhances predictive
performance but also offers significant insights into regional
disparities in respiratory risk. These findings stress the importance of
integrated environmental management and focused public health strategies
in addressing respiratory health issues.

# Code

All of the code can be found at this [GitHub
link.](https://github.com/abbbayyy/Capstone-Code.git)
