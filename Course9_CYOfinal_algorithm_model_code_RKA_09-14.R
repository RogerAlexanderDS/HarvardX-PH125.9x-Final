# HarvardX Data Science Course 9 - Capstone Project (Choose Your Own)

# "Machine Learning Time Series Forecasting for Solar Data Prediction"
# Author: Roger K. Alexander
# 2023-09-14
#
# This R script provides the code for the final "Choose Your Own" algorithm developed as part of the 
# Data Science Course 9 Capstone project. As discussed in the companion R Markdown document report,
# this program includes code for the download of data from a public website using their specified
# API and a private API key. The data set is therefore included as part of the project submission.
# 
# This program setups up and utilizes the caretForecast package to perform univariate forecasting of
# a particular solar radiation parameter. In addition to the use of multiple individual ML models for
# data forecasting, the code creates functions for an introduced Mean Absolute *Daily* Percentage
# Error (MADPE) performance metric and also setups and derives prediction results for a number of 
# ensemble models.
#
# Please refer to the companion Rmarkdown document and report for details on the analysis setup and 
# execution supported by this R script program.
#
# Due to the data size and the memory and processing that is required for the project analysis, the
# following R script code includes the incorporation of a scaling parameter called "scaler" that is 
# specifically et to allow for significant data compression that will allow the program to run in 
# minutes rather than hours. The value of that scaling parameter is currently set to 12 in this 
# R script (and set at 2 for the R Markdown report).

# Install required packages if not already locally installed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidyselect)) install.packages("tidyselect", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(httr)) install.packages("httr", repos = "http://cran.us.r-project.org")
if(!require(httr2)) install.packages("httr2", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tibble)) install.packages("tibble", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")
if(!require(formatR)) install.packages("formatR", repos = "http://cran.us.r-project.org")
if(!require(skimr)) install.packages("skimr", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(tseries)) install.packages("tseries", repos = "http://cran.us.r-project.org")
if(!require(zoo)) install.packages("zoo", repos = "http://cran.us.r-project.org")

# ML algorithms and related packages
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(caretForecast)) install.packages("caretForecast", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(imputeTS)) install.packages("imputeTS", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(brnn)) install.packages("brnn", repos = "http://cran.us.r-project.org")
if(!require(elasticnet)) install.packages("elasticnet", repos = "http://cran.us.r-project.org")

# Force install tinytex - a custom LaTeX distribution for supporting Rmd pdf reports (alternate to above)
# tinytex::install_tinytex(force=TRUE)

# Load the required libraries
library(tidyverse)
library(tidyselect)
library(ggplot2)
library(lubridate)
library(knitr)
library(httr)
library(httr2)
library(readr)
library(dplyr)
library(tibble)
library(stringr)
library(tinytex)
library(formatR)
library(skimr)
library(broom)
library(tseries)
library(zoo)
library(caret)
library(caretForecast)
library(caretEnsemble)
library(imputeTS)
library(glmnet)
library(brnn)
library(elasticnet)


#####################################################################################################
# START: Modules for NREL NSRDB Database API Implementation and for Data Set Download and Compilation 
#####################################################################################################

# *** ALL OF THE CODE RELATED TO THE DATA SET DOWNLOAD HAS BEEN COMMENTED OUT
# *** THE DATA SET NEEDED FOR THIS PROJECT HAVE BEEN PROVIDED AS CSV FILES
# *** THAAT MUST BE COPIED TO YOUR LOCAL WORKING DIRECTORY

# This API implementation also requires a private API key to access the NREL NSRDB site. An individual
# key can be requested at https://developer.nrel.gov/signup/ 

# Note: this process could take several minutes depending on the number of years of data downloaded
# and the number of parameters requested. In addition there are introduced time delays (of a 3 seconds) 
# between individual file downloads to ensure that NREL NSRDB per-user data download volume limits
# are not exceeded.

################### Code for NSRDB Solar Radiation Data File Downloads #########
# The NREL NSRDB data files are downloaded and processed in accordance with the API data format
# specification. Based on the format in which the data is provided a further processing step is taken to
# produce the final file version that is included as part of the project submission. This wrangling of the
# data ensures that a simple named data frame is available for project review and analysis.

# # Function for implementing NSRDB download using specified input of locations, years, and attributes
# nsrdb_download <- function(locations, years, attributes, interval){
#   locations <- locations
#   years <- years
#   attributes <- attributes
#   interval <- interval
#   
#   # Initialize an empty list to store downloaded data
#   all_data <- list()
#   
#   # Looping through locations and years to download data
#   for (loc in locations) {
#     loc_data <- list()
#     loc_data_hdr <- list()
#     
#     for (year in years) {
#       # Specifying API parameters for year data file download
#       lat <- loc[1]
#       lon <- loc[2]
#       leap_year <- 'false' # flag available to provide data with uniform, non-leap year cycles
#       year <- year
#       utc <- 'false'
#       your_name <- 'Roger+Alexander'
#       your_email <- 'rogeralexar@gmail.com'
#       your_affiliation <- ''
#       mailing_list <- 'false'
#       reason_for_use <- 'research'
#       api_key <- api_key
#       attributes <-  attributes
#       url <- sprintf("https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT(%f%%20%f)&names=%d&leap_day=%s&interval=%s&utc=%s&full_name=%s&email=%s&affiliation=%s&mailing_list=%s&reason=%s&api_key=%s&attributes=%s",
#                      lon, lat, year, leap_year, interval, utc, your_name, your_email, your_affiliation, mailing_list, reason_for_use, api_key, attributes)
#       
#       # Downloading year data
#       data_file <- read.csv(url)
#       
#       # Processing year data file to create time series metric data frame
#       yr_data <- as.data.frame(data_file) %>%
#         select(1:6) %>%
#         filter(row_number() > 1)
#       
#       names(yr_data) <- yr_data[1,]
#       yr_data <- yr_data[-1,]
#       
#       # Converting csv data file to tibble of numeric values
#       yr_data <- apply(yr_data, 2, parse_number)
#       yr_data <- as_tibble(yr_data)
#       yr_data
#       
#       # Processing year data file to get meta data tags (including parameter units)
#       meta <- as.data.frame(data_file) %>%
#         filter(row_number() == 1)
#       meta <- as_tibble(meta)
#       meta
#       
#       # add year data label to location list compiled for given location
#       loc_data[[as.character(year)]] <- yr_data
#       
#       # Adding meta info tag to location data header list compiled for each given location
#       loc_data_hdr[[as.character(year)]] <- meta
#       
#       # Inserting a system sleep before requesting next year data file
#       Sys.sleep(3) # time in seconds 
#       # NREL limits (1 request per second as well as total downloads per day 5000 for csv files)
#       # https://developer.nrel.gov/docs/solar/nsrdb/guide/#:~:text=The%20API%20is%20restricted%20in%20several%20ways%20including,2%20seconds%2020%20requests%20in%20process%20at%20once
#     }
#     # Adding location data year files and meta data year files to aggregated all_data list
#     all_data[[paste("LOC", str_sub(loc[1],1,6), str_sub(loc[2],1,6), sep = "_")]] <- loc_data
#     all_data[[paste("LOC", str_sub(loc[1],1,6), str_sub(loc[2],1,6), "meta", sep = "_")]] <- loc_data_hdr
#   }
#   # Aggregated data output
#   all_data
# }
# 
# # Using the above function, the NSRDB solar data files can be downloaded and compiled into a list structure. 
# # The created contents list can be reviewed and is then parsed using a second function to create the named
# # csv data frame files for the project submission.
# 
# # Reading (individual private) API key maintained in local text file
# api_key <- readLines("api_key.txt")
# 
# # Specifying function argument for download of solar data file
# 
# # Specifying a list of locations in (latitude, longitude) pairs
# # The NSRDB api allows the use of well-known-text (WKT) specification to provide a shape rather than point 
# # location with data automatically downloaded for multiple locations associated with the WKT shape.
# # The current code uses a list to allow multiple individual point downloads
# locations <- list(
#   #c(1.492384, 10.170104), # Equatorial Guinea
#   #c(42.248939, -101.95766), # Alkali Lake, NE
#   c(40.7128, -74.0060))  # New York, NY
# 
# # Specifying range of years for data download. With NSRDB access api, data is downloaded one year at a time 
# # option for multiple consecutive years file download
# years <- seq(2000, 2020)
# 
# # Setting the attributes of required variables (e.g., ghi, dhi, etc.) as a single comma-separated character.
# attributes <-  'dni'
# 
# # Setting the interval for the downloaded data specifying whether data is provided on an hourly (60)
# # or half-hourly (30) basis
# interval <- 60 # 
# 
# data_download <- nsrdb_download(locations, years, attributes, interval)
# 
# # Overview of downloaded file contents
# data_download
# str(data_download)
# 
# # Reviewing downloaded data file contents
# str(data_download[[1]])
# head(data_download[[2]])
# 
# # Downloading example data file with multiple attributes and data for a single year, 2020.
# # Specify function input arguments for download of example data file
# ex_locations <- list(c(40.7128, -74.0060))
# ex_years <- 2020
# ex_attributes <- 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
# ex_interval <- 60
# 
# example_file <- nsrdb_download(ex_locations, ex_years, ex_attributes, ex_interval)
# str(example_file)
# 
# # Writing example_file to local directory
# path_now <- getwd()
# write.csv(example_file, paste0(path_now, "/example_file.csv"), row.names=FALSE)
# 
# 
# ######## Aggregating Solar Data from downloaded data files list (including downloaded meta data)
# 
# # Function for extracting and compiling years of downloaded solar radiation data
# yrs_locdata_filecompile <- function(location, years, data_list){
#   compiled_data <- tibble()
#   
#   # set labels as used in the NSRDB file download for the targeted file location
#   file_label <- paste("LOC", str_sub(location[1],1,6), str_sub(location[2],1,6), sep = "_")
#   meta_label <- paste("LOC", str_sub(location[1],1,6), str_sub(location[2],1,6), "meta", sep = "_")
#   
#   for(year in years){
#     year <- as.character(year)
#     yr_data <- data_list[[file_label]][[year]]
#     yr_meta <- data_list[[meta_label]][[year]]
#     if(year == yrs[1]) {
#       loc_check <- yr_meta$Location.ID
#     }
#     meta_data <- yr_meta
#     locid <- yr_meta$Location.ID
#     ifelse(locid == loc_check,
#            compiled_data <- bind_rows(compiled_data, yr_data),
#            print("DATA INCONSISTENT - Year Data Not Added"))
#     return_data <- list(compiled_data, meta_data)
#   }
#   return_data
# }
# 
# ############## Code for preparing source data file for inclusion with project submission ##################
# # Specifying particular location (latitude, longitude) to be extracted for time-series solar data compilation
# loc <- c(40.7128, -74.0060)
# 
# # Specifying downloaded years of data from which the solar data file will be compiled
# # (this can be a subset of the downloaded data)
# yrs <- seq(2000, 2020)
# 
# # Creating solar data file by running compilation function on downloaded data list
# solar_data <- yrs_locdata_filecompile(loc, yrs, data_download)[[1]]
# meta_data <- yrs_locdata_filecompile(loc, yrs, data_download)[[2]]
# 
# # Extracting the NREL NSRDB Unique Location ID
# locid <- meta_data$Location.ID
# 
# # Creating file name label with assigned meta data location ID and downloaded data years range
# file_label <- paste0("solar_data_LOCID", "_", locid, "_", min(yrs), "-", max(yrs), ".csv")
# 
# # Creating metadata file name label with assigned location ID and downloaded data years range
# metafile_label <- paste0("solar_data_LOCID", "_", locid, "_", min(yrs), "-", max(yrs), "_meta.csv")
# 
# # Writing compiled solar data csv data file to local directory file
# path_now <- getwd()
# write.csv(solar_data, paste0(path_now, "/", file_label), row.names=FALSE)
# write.csv(meta_data, paste0(path_now, "/", metafile_label), row.names=FALSE)
# 
# # For verification that the data preparation function will work correctly for the project reviewer
# # the following code is used to read back the downloaded and prepared data files.
# 
# # Reading compiled solar data csv data file from local directory file for analysis
# path_now <- getwd()
# 
# # File name and path for retrieving and reading in source data file
# local_source_file <- paste0(path_now, "/", file_label)
# local_source_metafile <- paste0(path_now, "/", metafile_label)
# 
# source_data <- read.csv(local_source_file)
# source_metadata <- read.csv(local_source_metafile)
# 
# # Reviewing retrieved file contents
# head(source_data)
# head(source_metadata)
# 
###################################################################################################
# END: Modules for NREL NSRDB Database API Implementation and for Data Set Download and Compilation 
###################################################################################################


# Calculating MADPE - Mean Absolute Daily Percentage Error used as additional forecasting performance assessment 
# reference

# This function takes as input the actual and the forecasts measurements and the number of measurement points
# per day given by the daily_points parameter (as specified for the defined time series data). 
# In the case of hourly data, the measurements are aggregated into 24-point day groups.
# For each 24-hour day, the absolute value of the forecast error for each period of the day is calculated and the 
# sum of these errors used to derive the total daily error as a percentage of the total cumulative daily DNI.
 
MADPE <- function(true_measurements, forecast_measurements, daily_points){
  dat_df <- data.frame(actual=true_measurements, forecast=forecast_measurements)
  madpe <- dat_df %>%
    mutate(Abs_Error= abs(actual-forecast), DayNumber = (row_number()-1) %/% daily_points) %>%
    group_by(DayNumber) %>%
    summarize(adpe = 100*sum(Abs_Error)/sum(actual)) %>%
    pull(adpe) %>%
    mean()
}

############ Initial Data Set Retrieval and Exploration #########################

# !!! IMPORTANT !!!
# Please copy the downloaded the .csv data set files ('source_file' and 'example_file') to your current R 
# working directory given by the path obtained from the getwd() function call

# Once the csv files are copied to the working directory the R script will access them from there with the 
# following code that uses the relative local path addresses for the file access

# Determine your current R working directory
getwd()

# Set the path object to the current working directory
path_wd <- getwd()

# These file labels are the names of the csv data file provided for review
datafile_label <- "solar_data_LOCID_1244690_2000-2020.csv"
metafile_label <- "solar_data_LOCID_1244690_2000-2020_meta.csv"

# File name and path for retrieving and reading in source data file
local_source_filepath <- paste0(path_wd, "/", datafile_label)
local_source_metafilepath <- paste0(path_wd, "/", metafile_label)

# Reading in source file for analysis (plus metadata file) 
source_data <- read.csv(local_source_filepath)
source_metadata <- read.csv(local_source_metafilepath)

# Object for solar measurements metadata from the input source file
solar_metadata <- source_metadata

# Displaying metadata contents from the input source file (including the parameter units used for 
# measurement and environmental variables available from NSRDB)
str(solar_metadata)

# Note: The NREL NSRDB API that was used for the download of individual location solar data includes
# metadata in each (date-defined) CSV data file file as well as the ability to specify the set of solar
# radiation measurement attribute information that is downloaded. 
# In downloading and compiling the input data set for this project, the metadata is separated from the 
# measurement data but maintained as part of the compiled data list containing the aggregated data files. 

# Object for solar measurements solar data set from the input source data file
solar_datafile <- source_data

# Examining the structure and contents of the input data set
# Note: Since the undertaken analysis is based on univariate prediction, only a single attribute, the Direct 
# Normal Irradiance (DNI) is included in the compiled data set
str(solar_datafile)

# With hourly interval data and no leap year data (as explicitly specified in API database request), the
# expected number of data points for 21 years of data (maximum available for download at selected location) is:
24*365*21

# The oldest time period covered by the full data file is from January 1, 2000, as shown below at the head 
# of the data frame. 
# Note: The hourly data is provided on the half-hour mark (30 minute)
head(solar_datafile)

# As can be confirmed below, the most recent date-time in the data file is for Year=2020, Month=12, Day=31, 
# and Hour=23, as shown at the tail of the data frame.
tail(solar_datafile)


####### Data Set Date-Time Specification and Data Reference Specification

# For analysis purposes a date-time object is derived from the measurement time information provided in the 
# data set file. This Date object is then used to convert the DNI measurements into a time-series (ts) object. 
# Both the generated Date variable and the DNI ts object can be used for time referencing of the data set. 

# The Date variable can be used as the reference for monthly time slicing of the data set for analysis 
# and related time period referencing including data sets partitioning. 

solar_dataset <- solar_datafile %>%
  mutate(Date = ymd_hm(paste(Year, Month, Day, Hour, Minute,  sep= ' '), tz ="EST")) %>%
  as_tibble()

# Note: The explicitly time zone setting to "EST" (America/New_York) based on the NSRDB API download request 
# to provide local time rather than UTC time whereas the ymd_hm() defaults to a "UTC" time zone for 
# date-time conversions.

# Solar data set with date-time object reference variable added to data frame
solar_dataset
tail(solar_dataset)

# Ex. using Date variable to check on the solar data set time period

# Earliest measurement data point
min(solar_dataset$Date)

# Most recent measurement data point
max(solar_dataset$Date)


############ Converting DNI data to a time series object #######################

# From the solar radiation data set we create a time series object for the (hourly) DNI variable
dni_ts <- solar_dataset %>%
  mutate(dni_ts = ts(DNI, start = c(Year[1], 1), frequency = 8760)) %>%
  pull(dni_ts)

# The frequency argument specifies the maximum lag or cycle in the data given in data points
str(dni_ts)

# This time series object will be used as input to the caretForecast analysis package that will support
# the undertaken analyses


############ Preliminary Data Review ###########################################

# The skimr package provides a capability to readily show key descriptive stats for the data frame information. 
# In this case the skim() function can be used to verify the DNI variable data and transformed Date 
# completeness and display numeric stats including DNI value extremities. 

# A few normally interesting descriptive statistics that can be gleaned from a data frame of numeric data
data_skim <- skim(solar_dataset)
data_skim %>%
  kable()

# While the skim() function is a useful one for quick preliminary data reviews, it is less interesting when
# dealing with a single numeric variable and date-time object.


############ Data Content Exploration ###################

############ Plotting DNI Variation on a single day
# Taking a randomly selected day from the data set for data exploration

# Randomly selected day
s_day <- sample(solar_dataset$Date, 1)

solar_dataset %>%
  filter(Year== year(s_day) & Month==month(s_day) & Day == day(s_day)) %>%
  mutate(Hour = hour(Date)+ minute(Date)/60) %>%
  ggplot(aes(x=Hour, y=DNI)) + 
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(0,23, 2)) +
  ggtitle("Daily Direct Normal Irradiance (DNI) Variation - Single Day", as.Date(s_day)) + labs(x = "Hour", y = "DNI (w/m2)")

# Randomly selected day
s_day <- sample(solar_dataset$Date, 1)

solar_dataset %>%
  filter(Year== year(s_day) & Month==month(s_day) & Day == day(s_day)) %>%
  mutate(Hour = hour(Date)+ minute(Date)/60) %>%
  ggplot(aes(x=Hour, y=DNI)) + 
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(0,23, 2)) +
  ggtitle("Daily Direct Normal Irradiance (DNI) Variation - Single Day", as.Date(s_day)) + labs(x = "Hour", y = "DNI (w/m2)")

# The number of hours of positive sunlight also depends on the time of the year for the given location - 
# New York in this data set.

############ Plotting Daily DNI Variation - Monthly Averages

# Randomly selected year
s_year <- year(sample(solar_dataset$Date, 1))

solar_dataset %>%
  filter(Year == s_year) %>%
  mutate(Hour = hour(Date)+ minute(Date)/60,
         Calendar_Month = factor(month.name[month(Date)], levels = month.name)) %>%
  group_by(Month, Calendar_Month, Hour) %>%
  summarize(DNI_Hourly_Avg = mean(DNI)) %>%
  ggplot(aes(x=Hour, y=DNI_Hourly_Avg, group = desc(Calendar_Month))) + 
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(0,23, 2)) +
  facet_wrap(~Calendar_Month) +
  ggtitle("Daily Direct Normal Irradiance (DNI) Variation - Monthly Averages", s_year) + 
  labs(x = "Hour", y = "DNI (w/m2) - Monthly Average")


############ Plotting Daily DNI Totals

# Randomly selected year
s_year <- year(sample(solar_dataset$Date, 1))

solar_dataset %>%
  filter(Year == s_year) %>%
  mutate(Hour = hour(Date)+ minute(Date)/60,
         Calendar_Month = factor(month.name[month(Date)], levels = month.name)) %>%
  group_by(Month, Calendar_Month, Day) %>%
  summarize(DNI_Daily_Total = sum(DNI)) %>%
  ggplot(aes(x=Day, y=DNI_Daily_Total, group = desc(Calendar_Month))) + 
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(2,30, 4)) +
  facet_wrap(~Calendar_Month) +
  ggtitle("Total Daily Direct Normal Irradiance (DNI) - By Month") + 
  labs(x = "Day", y = "Total Daily DNI (w/m2)") +
  theme(axis.text.x = element_text(angle = 0, vjust = 0, hjust=0, size=8))


############ Decomposing Trends and Cycles in the DNI Hourly Time Series Data

# Sub-setting first three years of data set DNI time series data 
dni_ts_3yr <- window(dni_ts, start = c(year(solar_dataset$Date[1]), 1), 
                     end = c(year(solar_dataset$Date[1])+2, 8760) )

# Apply decomposition function and plotting decomposed data components
dni_dcomp <- decompose(dni_ts_3yr)

plot(dni_dcomp)


############ Decomposing Trends and Cycles in the DNI Hourly Values - Monthly Averaged Time Series Data

# Defining 3-year period from first year (2000) in the data set
year_range <- seq(year(solar_dataset$Date[1]), year(solar_dataset$Date[1])+2)

# Filtering 3-year period and calculating monthly average DNI values
dni_avgs_3yr <- solar_dataset %>%
  filter(Year %in% year_range) %>%
  mutate(Year = year(Date), Hour = hour(Date)+ minute(Date)/60,
         Calendar_Month = factor(month.name[month(Date)], levels = month.name)) %>%
  group_by(Year, Month, Calendar_Month, Hour) %>%
  summarize(DNI_Hourly_Avg = mean(DNI))

# Creating DNI hourly averaged time series with (24*12)= 288 samples per year
# Note the need to coerce group_tibble to tibble to ensure that time series object is obtained.
dni_avgs_3yr_ts <- as_tibble(dni_avgs_3yr) %>%
  mutate(dni_avgs_ts = ts(DNI_Hourly_Avg, start = c(Year[1], 1), frequency = 288)) %>%
  pull(dni_avgs_ts)

# class(dni_avgs_ts)
# str(dni_avgs_ts)

# Apply decomposition function and plotting decomposed data components
dni_avgs_3yr_dcomp <- decompose(dni_avgs_3yr_ts)

plot(dni_avgs_3yr_dcomp)


############ Partitioning Data into Training and Testing Data Sets #############

# The training and evaluation testing data sets are given by the following:

test_set <- window(dni_ts, start = c(2020, 1)) # validation testing data set
length(test_set)

train_set <- window(dni_ts, start = c(2000, 1), end = c(2019, 8760)) # training data set
length(train_set)

# Additionally given that we know the validation test_set will be 26280 hourly DNI data samples,
# the caretForecast package can also be used to perform the time series data split. The validation test_set
# is automatically selected as the final consecutive data samples given by the specified test_size argument.
# The training train_set is the remainder of the data set samples prior to the start of the test_set.

# Using caretForect data splits
datlists <- caretForecast::split_ts(dni_ts, test_size = 8760)

test_set <- datlists$test
length(test_set)

train_set <- datlists$train
length(train_set)

# Both the test and train data sets retain their time series properties including time series frequency
head(test_set) 
# Note: End is just the 6th sample of the year since head() only provides the top 6 values of the data set

tail(train_set)
# Note: Start is the 8755th sample of the year since tail() only provides the bottom 6 values of the data set

# Further verifying the data set split by combining the two time series
data_sets_ts <- ts(c(train_set, test_set), start = start(train_set), frequency = frequency(train_set))

# Comparing with the original ts data set
identical(data_sets_ts, dni_ts)


############ Reducing Analysis Data Set Size ####################################

# To continue with the analysis with the limited processing capability of a standard laptop computer
# the solar radiation time series data is compressed to monthly daily averages. With this approach,
# rather than having (365*24) 8760 hourly DNI samples per year, there will just be (12*24) 288 hourly
# DNI samples per year (24 monthly average hourly samples per month). This reduces the data size by a
# factor of ~30. 

############ Calculating monthly averaged daily DNI values
dni_avgs <- solar_dataset %>%
  mutate(Year = year(Date), Hour = hour(Date) + minute(Date)/60,
         Calendar_Month = factor(month.name[month(Date)], levels = month.name)) %>%
  group_by(Year, Month, Calendar_Month, Hour) %>%
  summarize(DNI_Hourly_Avg = mean(DNI))

# Creating DNI hourly averaged time series with (24*12)= 288 samples per year
# Note the need to coerce group_tibble to tibble to ensure that time series object is obtained.
dni_avgs_ts <- as_tibble(dni_avgs) %>%
  mutate(dni_avgs_ts = ts(DNI_Hourly_Avg, start = c(Year[1], 1), frequency = 288)) %>%
  pull(dni_avgs_ts)

class(dni_avgs_ts)
str(dni_avgs_ts)


############ Compressed Data Sets ##############################################

test_set <- window(dni_avgs_ts, start = c(2019, 1)) # validation testing data set
length(test_set)

train_set <- window(dni_avgs_ts, start = c(2000, 1), end = c(2018, 288)) # training data set
length(train_set)

# Training data set for input to forecasting model
training_data <- train_set
length(train_set)

# Testing (holdout evaluation) data set for use in forecasting with generated algorithm models
testing_data <- test_set
length(test_set)


############ Aggregating Time Series Data to Further Reduce Processing Time ####

# The 'scaler' parameter is introduced to allow more automated control of the compressing of the data 
# set used in the analysis. The scaler controls how many points in the daily time series measurements are 
# aggregated from the original baseline of 24 hourly points per day.


############ Further Compressed Data Set #######################################

scaler <- 12 # number of 1-hour periods per day that are aggregated (as sub-multiple of 24)

# dni_comp_ts is the new multiple-hour aggregated time series (where per day hourly aggregation is added 
# to the prior monthly averaged data)
dni_comp_ts <- aggregate(dni_avgs_ts, 288/scaler, sum)

# With the further data compression a 2-year test data set is defined
test_set <- window(dni_comp_ts, start = c(2019, 1)) # validation testing data set
length(test_set)

# With the further data compression, the training set duration is further extended to use more of the 
# available data set
train_set <- window(dni_comp_ts, start = c(2006, 1), end = c(2018, 288/scaler)) # training data set
length(train_set)


############ Model Setup and Analysis ##########################################

# Automated setup and cross-validation tuning capabilities of the caretForecast package are used as the 
# primary tool for the analysis.

############ Using caretForecast forecasting package ###########################

# The most up-to-date caretForecast package is available from GitHub rather than the CRAN repo. To ensure
# a version compatible with the latest R versions, the following code is included. 
# Note: It may be necessary to answer a number of questions on related package components to complete the 
# installation.

# if(!require("devtools")) install.packages("devtools", repos = "http://cran.us.r-project.org")
# devtools::install_github("Akai01/caretForecast")


############ Single Model Time Series Evaluation ###############################

set.seed(100, sample.kind = "Rounding") 

training_data <- train_set
testing_data <- test_set

# This print code provides visual feedback and time logging for the on-going analysis code execution
print(paste0(" Initiating Model Evaluation: Train_set data size : ", length(training_data), " data points"))
start_time <- Sys.time()
print(paste0("Model Evaluation Start time: ", Sys.time()))

# Using caretForecast (Conformal Time Series Forecasting Using State of Art Machine Learning Algorithms)

# Given the range of parameters that must be tuned for each model, the execution time can be quite
# significant when CV is applied. To gauge approximate execution times for a given model, some initial
# exploratory testing was conducted with CV disabled. Since resampling is also automated as part of the 
# caret cross validation, the duration of the total training data set also affects execution time since with
# time series data needing to be continuous, the CV samples are shifted as a function of the max lag time
# (which in this work is specified as 1 year - 288 hourly monthly-averaged samples for scaler = 1). 
# This time shifting of resamples limits how many time-shifted resamples can be drawn from the available 
# training data set.

#### Prior to running check scaler parameters in accordance with expected execution time!!

daily_points <- 24/scaler
# 'scaler' parameter determines the number of hourly measurements that are combined

lag <- 288/scaler 
# 288 corresponds to a 1-year period of 12 months x 24 monthly-averaged hourly points per day profile

window <- 1728/scaler 
# 1728 (288*6) corresponds to specified 6-year model training period (per resample) for 2-year forecasts

horizon <- 576/scaler # 576 corresponds to a 2-year (monthly-averaged data) test period for CV samples

# Testing execution time without CV
fit_cub <- ARml(training_data, max_lag = lag, initial_window=window, caret_method = "cubist",
            cv = TRUE, cv_horizon=horizon, fixed_window=TRUE, metric="RMSE", verbose = FALSE)

# > Note: The max_lag argument is the maximum period over which data repetition cycles occur. Since the 
# solar data has annual cycles, that corresponds to a max_lag of 24x365 = 8760 hourly data points (using 
# the specified NSRDB download API to explicitly remove leap year data).
# 
# > The fixed_window=TRUE uses fixed size training-testing data sets for cross-validation (cv=TRUE)
# 
# > The initial_window argument is the number of consecutive values in each training set sample. 
# Since the goal of the analysis is to train on fixed 8-year training data, that corresponds to 
# 6x24x365 = 52560 hourly values (before monthly averaging compression).
# 
# > The cv_horizon argument is the number of consecutive values in the cv test set sample. Since the 
# analysis goal is to perform 2-year forecasts, that corresponds to 2x24x365 = 17520 hourly values 
# (before monthly averaging compression).
# 
# > The metric argument specifies what summary metric will be used to select the optimal model. 
# By default, possible values are "RMSE" and "Rsquared" for regression.
# 
# > Note: For configured cross-validation with fixed window=TRUE, the following relationship must hold: 
# length(training_data) - length(initial_window), MUST be an integer of (max_lag)!!
#   
# > The single model analysis applies the "cubist" method, a rule-based prediction regression model that 
# is an extension of Quinlan’s M5 model tree https://cran.r-project.org/web/packages/Cubist/Cubist.pdf.

# Recording and displaying execution run time on-screen for visual confirmation and processing time
# benchmarking when setting up different models and evaluation.
stop_time <- Sys.time()
exec_time <- difftime(stop_time, start_time, units = "mins")
print(paste0("Model Evaluation executiion time: ", round(exec_time,1), " minutes"))

# With the above setup and the respective training and testing data, the forecast values, fct and cv-tuned
# accuracy measures are as given below where the level is the specified forecast confidence level:

# Using the single model generated data
forecast(fit_cub, h = length(testing_data), level = 95)-> fc

# Forecast performance for caretForecast CV-optimized model 
accuracy(fc, testing_data) %>%
  kable()

# Note that MPE and MAPE measures can be -Inf or Inf as a result of zero values

# Output plot of training and associated forecast data. Note the automated generation of confidence intervals
# on the forecast plots based on the 95% levels specified.
ggplot2::autoplot(fc) 

ggplot2::autoplot(fc) +
  labs(y = "Monthly-averaged Daily DNI W/m-2") +
  coord_cartesian(xlim = c(2016, 2021)) +
  scale_x_continuous(breaks = c(seq(2016, 2021)), minor_breaks = c(seq(2016, 2021, by=1/12) - 1/24) ) +
  theme(panel.grid.major.x = element_line(color="grey", linewidth =  0.5)) 

# Notes on forecast objects and time: 
# fc$x gives the training data set of the output forecast object
# fc$mean gives the mean of the forecast variable. Levels(upper, lower) provide confidence level values.
# class of fc$x and fc$mean is a time series object
# Time, as a decimal, from the time series object can be obtained with the function time("ts object") where
# the time series frequency determines the decimal increment!
# The input to the ggplot function when the forecast object is referenced is the decimal time for the time series
# given by time(fc$x + fc$mean). 

# Note: Decimal time can be converted to year-month using as.yearmon(time(fc$x))

ggplot2::autoplot(fc) +
  labs(y = "Monthly-averaged Daily DNI W/m-2") +
  coord_cartesian(xlim = c(2016, 2021)) +
  scale_x_continuous(breaks = c(seq(2016, 2021)), minor_breaks = c(seq(2016, 2021, by=1/12) - 1/24),
                     labels = as.yearmon(seq(2016, 2021))) +
  theme(panel.grid.major.x = element_line(color="grey", linewidth =  0.5)) 


# The Mean Absolute Daily Percentage Error MADPE measure can be calculated based on the defined MADPE function
madpe_accuracy <- data.frame(MADPE=MADPE(testing_data, fc$mean, daily_points))
rownames(madpe_accuracy) <- c("Test set")

madpe_accuracy %>%
  kable()

# Note since we are using the hourly DNI values (averaged per month), the MADPE is calculated over 24 hourly 
# points per day

# The result is thus a DNI daily average percentage error of 
print(paste0(round(madpe_accuracy[1,1], 2), " %"))


############ Analysis of Ensemble Models for DNI Forecasting ###################

set.seed(1, sample.kind = "Rounding") 

training_data <- train_set
testing_data <- test_set

models <- c("cubist", "glm", "ranger", "ridge", "svmLinear2")

fcts <- lapply(models, function(model){ 
  
  # This print code provides visual feedback and time logging for the on-going analysis code execution
  print(paste0("Initiating Model Evaluation: Train_set data size : ", length(training_data), " data points"))
  start_time <- Sys.time()
  print(paste0("Model Evaluation Start time: ", Sys.time()))
  
  print(model) # visual feedback while running
  
  # Since the "cubist" model has been previously run the ifelse() statement allows the prior "fit_cub" model 
  # data to be used when that model is called
  
  ifelse(model == "cubist", fit <- fit_cub,
         fit <- ARml(training_data, max_lag = lag, initial_window=window, caret_method = model,
              cv = TRUE, cv_horizon=horizon, fixed_window=TRUE, verbose=FALSE))
  
  forecast(fit, h = length(testing_data)) -> fc
  
  # Extracting forecast values
  dni_fct <- fc$mean
  
  # Extracting RMSE from the different output accuracy performance measures
  rmse_fct <- accuracy(fc, testing_data)['Test set',"RMSE"]
  
  # Calculating the project defined MADPE (mean absolute daily percentage error)
  madpe_fct <- MADPE(testing_data, fc$mean, daily_points)
  
  # Recording and displaying execution run time on-screen for visual confirmation and processing time
  # benchmarking when setting up different models and evaluation.
  stop_time <- Sys.time()
  exec_time <- difftime(stop_time, start_time, units = "mins")
  print(paste0("Model Evaluation executiion time: ", round(exec_time,1), " minutes"))
  cat("  \n")
  
  return(list(dni_fct, rmse_fct, madpe_fct))
}) 

# Set names for forecasts output by model used
names(fcts) <- models

# Verify forecasts output results
str(fcts)
fcts[1]

# Extracting the lists of forecast dni values
dni_fcts <- sapply(fcts, function(n){
  n[[1]]
})

# Extracting the lists of MAPE performance metric for each forecast by model used
rmse_fcts <- sapply(fcts, function(n){
  n[[2]]
})

# Extracting the lists of MADPE performance metric for each forecast by model used
madpe_fcts <- sapply(fcts, function(n){
  n[[3]]
})

# Creating data frames of dni forecasts by model and MAPE by model
dni_fcts_df <- data.frame(dni_fcts)
head(dni_fcts_df)

madpe_fcts_df <- data.frame(MADPE=madpe_fcts)
madpe_fcts_df


############ Output Results of applied single ML Models ########################

results_fcts_df <- data.frame(model=rownames(madpe_fcts_df), RMSE=rmse_fcts, MADPE=madpe_fcts)
rownames(results_fcts_df) <- seq(nrow(madpe_fcts_df))

# MADPE-ordered results on ML models
results_fcts_df %>%
  arrange(MADPE) %>%
  kable()


############ Removal of Forecast Models with Performance below Threshold #######

# Implementatio of a MADPE performance cutoff to keep the better performing forecasting algorithms
# This avoids polluting the ensembles for models that are seen as outlier-performers

filtered_madpe_fcts_df <- results_fcts_df %>%
  filter(MADPE < 20) # This 20% MADPE value was based arbitrarily chosen to remove appreciably higher MADPEs

# Results for models with MADPE better (lower) than 20% daily average percentage error
# (Table is only output if there have been models removed)

if(nrow(filtered_madpe_fcts_df) < nrow(results_fcts_df)){
  filtered_madpe_fcts_df %>%
    arrange(MADPE) %>%
    kable()
}

# Identifying any removed models
unfiltered_models <- intersect(results_fcts_df$model, filtered_madpe_fcts_df$model)

# Filtering the dni forecasts data frame based on common (un-removed) models
filtered_dni_fcts_df <- dni_fcts_df[c(unfiltered_models)]


############ Creating Ensemble Models ##########################################

# Two types of ensemble models are created. 
# The first is a combined average of the previous ML (sub) model results where an ML model is used to 
# combine the individual regression results. 

# Many choices exist. Testing of smaller sample iterations used to find select a model that provides 
# a good ensemble performance.

# Note: the search for a "best" model has not been exhaustive and instead simply compares the execution
# time and MADPE performance for a number of regression model candidates.

# The second approach to creating an ensemble model is through ensemble averaging. In this ensemble model, the mean
# median and a weighted average is used to create the average ensemble models.

###### Creating ML-Ensemble Model based on Regression Combining of sub-model forecasts
#
# A combined ensemble model is created by regression combining the sub-model forecast results to produce
# a fitted forecast. The MADPE measure for this combined ensemble forecast DNI is assessed and later also
# compared together other sub-models and ensemble models.

# The time series time variable is explicitly added to the data frame of sub-model forecasts and the data long-pivoted.
# Since all model forecasts are based on the same time series range, time is from the first model in the forecast
# list "fcts".

dni_ml_ens_df <- dni_fcts_df %>%
  mutate(time = as.numeric(time(fcts[[1]][[1]])) )

# Pivoting longer to create combined data frame of sub-model values versus time
ens_data <- dni_ml_ens_df %>%
  pivot_longer(-time, names_to = "model", values_to = "dni_ml_ens")

# Verifying pivoted data frame
head(ens_data)

# Using "random forest" to combine the sub-model forecasts into ML-ensemble forecast
# Note: This has not been a researched choice but simply based on results obtained from small sample data testing
ens_fit <- ens_data %>%
  train(dni_ml_ens ~ time, data=., method ="rf")

# Creating ML-ensemble forecast data vector
dni_ensCombine <- as.vector(predict(ens_fit, newdata=dni_ml_ens_df))

# Verifying output result of ML-ensemble forecast
head(dni_ensCombine)
str(dni_ensCombine)

##### Creating Average Ensemble Models
#
# For the average ensemble, three different models are derived using the forecast results of the individual
# sub-models which are combined to create simple average, a median, and a MADPE-weighted average. 
# For the MADPE-weighted average the forecast MADPE is normalized relative to the best performing MADPE 
# forecast (i.e, the one with the lowest MAPE). This weighted average is only expected to produce a better 
# ensemble model forecast only if there is some absence of correlation between the sub-model results that
# are exploited by the ensemble combining.

### Weighted Average Ensemble
# Creating an inverse MAPE weighting where the weight for a forecast is a function of its MAPE performance
# relative to the best (lowest) performing model such that models with lower MAPE (lower forecast % error) 
# are assigned a higher weighting factor

# Calculating MADPE-based weighting factors
# Note: weights are based on the MADPE-filtered models (but ordered alphabetically)
ens_fctWeights_df <- filtered_madpe_fcts_df %>%
  mutate(inv_madpe_wght = ifelse(MADPE == 0, 1, min(MADPE)/MADPE), 
         wghts = inv_madpe_wght/sum(inv_madpe_wght)) %>%
  arrange(model) %>%
  select(model, MADPE, weights=wghts)

# > Note: The weighting formulation assigns an inverse weighting of 1 if a model has a 0 MADPE (zero error, 
# perfect forecast). This results in a 0 weighting for any other non-zero MAPE models (and results in an equal weighting
# if there are multiple models with perfect forecast).

# Verifying weight and sum of weights
# Ensemble MADPE weights (note the ordering is not changed)
ens_fctWeights_df %>%
  arrange(MADPE) %>%
  kable()

# Ensemble Weights vector
ens_fctWeights <- ens_fctWeights_df$weights
length(ens_fctWeights)

# Sum of Ensemble Weights (equal to unity)
sum(ens_fctWeights)

# Average Ensemble Forecasts (Ensemble Average, Ensemble Median, and Ensemble Weighted Mean)
dni_ensMean <- apply(filtered_dni_fcts_df, 1, mean)
dni_ensMedian <- apply(filtered_dni_fcts_df, 1, median)

# For weighted ensemble calculation it is essential that the order of the model weights are the same as the 
# filtered_dni_fcts_df data frame columns. Alphabetic ordering is used to match the imposed ordering
# used for the ensemble weights data frame ens_fctWeights_df
filtered_dni_fcts_df <- filtered_dni_fcts_df[ , order(names(filtered_dni_fcts_df))]

dni_ensWeightMean <- apply(filtered_dni_fcts_df, 1, weighted.mean, ens_fctWeights)

# Verifying Simple Mean Ensemble Average model
str(dni_ensMean)
head(dni_ensMean)

# Obtaining DNI measurements from testing data ts object for use in calculating MADPE for ensemble models
dni_actual <- as.vector(testing_data)
dni_actual

# Creating list of the All Ensemble DNI Forecasts (including Average Ensemble and Combined Regression Ensemble)
dni_ens_fcts <- list(ensMean=dni_ensMean, ensMedian=dni_ensMedian, ensWeightMean=dni_ensWeightMean, 
                     ensCombine_rf=dni_ensCombine) 
dni_ens_fcts

# Ensemble Forecasts data frame
dni_ens_fcts_df <- as.data.frame(dni_ens_fcts)
head(dni_ens_fcts_df)

# Performance Results of single ML and Ensemble models

# The different ensemble model results are combined into a single data frame and the MADPE function used to 
# derive the associated mean absolute daily percentage error metric for each ensemble model

# Combining all dni forecasts (individual ML model DNI forecasts and Ensemble model DNI forecasts) into single data frame 
dni_all_fcts_df <- cbind(dni_fcts_df, dni_ens_fcts_df)
head(dni_all_fcts_df)

# MADPE function is applied to calculate the MADPE for Ensemble DNI Forecasts

# Calculating MADPEs for Ensemble Model DNI Forecasts - output list
madpes_ens_lst <- lapply(dni_ens_fcts, FUN=MADPE, true_measurements=dni_actual, daily_points=daily_points)
madpes_ens_lst

# RMSE function is applied to calculate the RMSE values for Ensemble DNI Forecasts
# Standard R RMSE <- function(pred, obs), a function of the predictions and observations (actuals)

# Calculating RMSEs for Ensemble Model DNI Forecasts - output list
rmses_ens_lst <- lapply(dni_ens_fcts, FUN=RMSE, obs=dni_actual)
rmses_ens_lst

# Ensemble Model Forecast MAPEs and RMSEs assessment results data frame
rmses_ens_fcts <- t(as.data.frame(rmses_ens_lst))
rmses_ens_fcts

madpes_ens_fcts <- t(as.data.frame(madpes_ens_lst))
madpes_ens_fcts

# Combining and ordering model results
# ML-algorithm models and Ensemble Forecast results data frame

results_ens_fcts <- data.frame(rmses_ens_fcts, madpes_ens_fcts)
colnames(results_ens_fcts) <- c("RMSE", "MADPE")
results_ens_fcts

results_ens_fcts_df <- results_ens_fcts %>%
  mutate(model = rownames(.)) %>%
  as_tibble() %>%
  select(model=model, RMSE, MADPE)
results_ens_fcts_df

# Combined Single ML-Models and Ensemble Models RMSEs and MADPEs
all_results_df <- rbind(results_fcts_df, results_ens_fcts_df)
all_results_df

# Combined and MADPE-Ordered results 
madpe_all_results_df <- all_results_df %>%
  arrange(MADPE) %>%
  select(model=model, MADPE, RMSE)

madpe_all_results_df %>%
  kable()

# Combined and RMSE-Ordered results 
rmse_all_results_df <- all_results_df %>%
  arrange(RMSE) %>%
  select(model=model, RMSE, MADPE)

rmse_all_results_df %>%
  kable()


############ Using Best MADPE-Performing Forecast for Revisiting Exploratory Review

# As done with the earlier initial DNI data exploration, the same is applied for the forecasts 
# made with the 'best' performing ML algorithm.

############ Reviewing 'Best' Forecasting Model

# Best forecasting model based on MADPE results
best_model <- madpe_all_results_df$model[1]
best_model

# DNI forecast for 'best' model
dni_best_fct <- dni_all_fcts_df[, best_model]
head(dni_best_fct)

# Convert dni_best_fct to a time series
dni_best_fct_ts <- ts(dni_best_fct, start = start(test_set), frequency = frequency(test_set))

# Adding the Best Forecast DNI Values to Test_set which includes the associated Date and other date-related 
# information variables

# Take solar data set and subset by the Date???
dni_best_fct_vct <- as.vector(dni_best_fct_ts)
dni_best_fct_tsv <- (time(dni_best_fct_ts))

DNI_Hourly_Avg <- as.vector(dni_best_fct_ts)
DNI_Year <- as.vector(format(yearmon(time(dni_best_fct_ts)), "%Y"))
DNI_Month <- as.vector(month.name[as.numeric(format(yearmon(time(dni_best_fct_ts)), "%m"))])
DNI_Hour <- as.vector(scaler*(row_number((time(dni_best_fct_ts))) -1) %% (24/scaler) + (scaler/2))

dni_best_fct_df <- data.frame(DNI_Year, DNI_Month, DNI_Hour, DNI_Hourly_Avg)
head(dni_best_fct_df)


# First year of the forecast years
fs_year1 <- dni_best_fct_df$DNI_Year[1]

dni_best_fct_df %>%
  filter(DNI_Year == fs_year1) %>%
  mutate(Hour = DNI_Hour,
         Calendar_Month = factor(DNI_Month, levels = month.name)) %>%
  group_by(DNI_Month, Calendar_Month, Hour) %>%
  ggplot(aes(x=Hour, y=DNI_Hourly_Avg, group = desc(Calendar_Month))) + 
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(0,23, 2*scaler)) +
  facet_wrap(~Calendar_Month) +
  ggtitle(paste0(best_model,"-model *Forecast* Daily Direct Normal Irradiance (DNI) Variation"), fs_year1) + 
  theme(plot.title = element_text(size = 12, face = "bold")) +
  labs(x = "Hour", y = "DNI (w/m2) - Monthly Average")

# To represent the original data in the same aggregated hourly increments as that used in the analysis, 
# some manipulation of the Hour variable is required.

solar_dataset %>%
  filter(Year == fs_year1) %>%
  mutate(Hour = hour(Date)+ minute(Date)/60,
         Calendar_Month = factor(month.name[month(Date)], levels = month.name)) %>%
  group_by(Calendar_Month, Hour) %>%
  summarize(DNI_Hourly_Avg = mean(DNI)) %>%
  mutate(Hour = (Hour %/% scaler) + scaler/2) %>%
  group_by(Calendar_Month, Hour) %>%
  summarize(DNI_xHourly_Avg = sum(DNI_Hourly_Avg)) %>%
  ggplot(aes(x=Hour, y=DNI_xHourly_Avg, group = desc(Calendar_Month))) + 
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(0,23, 2*scaler)) +
  facet_wrap(~Calendar_Month) +
  ggtitle("*Actual* Daily Direct Normal Irradiance (DNI) Variation", fs_year1) + 
  theme(plot.title = element_text(size = 12, face = "bold")) +
  labs(x = "Hour", y = "DNI (w/m2) - Monthly Average")


# Second year of the forecast years
fs_year2 <- as.numeric(fs_year1) + 1

dni_best_fct_df %>%
  filter(DNI_Year == fs_year2) %>%
  mutate(Hour = DNI_Hour,
         Calendar_Month = factor(DNI_Month, levels = month.name)) %>%
  group_by(DNI_Month, Calendar_Month, Hour) %>%
  ggplot(aes(x=Hour, y=DNI_Hourly_Avg, group = desc(Calendar_Month))) + 
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(0,23, 2*scaler)) +
  facet_wrap(~Calendar_Month) +
  ggtitle(paste0(best_model,"-model *Forecast* Daily Direct Normal Irradiance (DNI) Variation"), fs_year2) + 
  theme(plot.title = element_text(size = 12, face = "bold")) +
  labs(x = "Hour", y = "DNI (w/m2) - Monthly Average")

# To represent the original data in the same aggregated hourly increments as that used in the analysis, 
# some manipulation of the Hour variable is required.

solar_dataset %>%
  filter(Year == fs_year2) %>%
  mutate(Hour = hour(Date)+ minute(Date)/60,
         Calendar_Month = factor(month.name[month(Date)], levels = month.name)) %>%
  group_by(Calendar_Month, Hour) %>%
  summarize(DNI_Hourly_Avg = mean(DNI)) %>%
  mutate(Hour = (Hour %/% scaler) + scaler/2) %>%
  group_by(Calendar_Month, Hour) %>%
  summarize(DNI_xHourly_Avg = sum(DNI_Hourly_Avg)) %>%
  ggplot(aes(x=Hour, y=DNI_xHourly_Avg, group = desc(Calendar_Month))) + 
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(0,23, 2*scaler)) +
  facet_wrap(~Calendar_Month) +
  ggtitle("*Actual* Daily Direct Normal Irradiance (DNI) Variation", fs_year2) + 
  theme(plot.title = element_text(size = 12, face = "bold")) +
  labs(x = "Hour", y = "DNI (w/m2) - Monthly Average")

