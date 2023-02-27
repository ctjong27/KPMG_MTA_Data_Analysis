# https://www.epa.gov/system/files/documents/2022-05/ctyfactbook2021.xlsx

library(tidyverse)
library(readxl)
library("stringr") # allows the use of "left" and "right" string manipulations

get_county_climate <- function() {
    if(!file.exists("get_county_climate.csv")) {
        
        # precipitation annual mean
        # https://www.ncei.noaa.gov/cag/county/mapping/110-pcp-202012-12.csv
        precip_annual_mean_df <- read.csv("https://www.ncei.noaa.gov/cag/county/mapping/110-pcp-202012-12.csv", skip = 3)
        colnames(precip_annual_mean_df)[6] <- "precip_century_annual_mean_1901_2000"

        # average temperature annual mean
        # https://www.ncei.noaa.gov/cag/county/mapping/110-tavg-202012-12.csv
        avg_temp_annual_mean_df <- read.csv("https://www.ncei.noaa.gov/cag/county/mapping/110-tavg-202012-12.csv", skip = 3)
        colnames(avg_temp_annual_mean_df)[6] <- "avg_temp_annual_mean_1901_2000"

        # maximum temperature annual mean
        # https://www.ncei.noaa.gov/cag/county/mapping/110-tmax-202012-12.csv
        max_temp_annual_mean_df <- read.csv("https://www.ncei.noaa.gov/cag/county/mapping/110-tmax-202012-12.csv", skip = 3)
        colnames(max_temp_annual_mean_df)[6] <- "max_temp_annual_mean_1901_2000"

        # minimum temperature annual mean
        # https://www.ncei.noaa.gov/cag/county/mapping/110-tmin-202012-12.csv
        min_temp_annual_mean_df <- read.csv("https://www.ncei.noaa.gov/cag/county/mapping/110-tmin-202012-12.csv", skip = 3)
        colnames(min_temp_annual_mean_df)[6] <- "min_temp_annual_mean_1901_2000"

        
        # state fips code
        state_fips_df <- read.csv("https://gist.githubusercontent.com/dantonnoriega/bf1acd2290e15b91e6710b6fd3be0a53/raw/11d15233327c8080c9646c7e1f23052659db251d/us-state-ansi-fips.csv")
        
        # Left joins are possible because each dataframe has 3141 rows
        df_lj <- left_join(precip_annual_mean_df, avg_temp_annual_mean_df, by = c("Location.ID"="Location.ID"))
        df_lj <- left_join(df_lj, max_temp_annual_mean_df, by = c("Location.ID"="Location.ID"))
        df_lj <- left_join(df_lj, min_temp_annual_mean_df, by = c("Location.ID"="Location.ID"))
        
        # # Testing retrieving left and right string statements
        # str_sub(df_lj$Location.ID, 1, 2)
        # str_sub(df_lj$Location.ID, -3, -1)
        # 
        # # filter works for individual numbers, but not for string acronyms, that requires grepl
        # state_fips_df %>% filter(st==5)
        # 
        # # Breaking down state acronym into fips state code
        # state_fips_df %>% filter(grepl('GA', stusps))
        # (state_fips_df%>%filter(grepl('GA', stusps)))['st']
        # 
        # df_lj['fips'] <- strtoi(paste((state_fips_df%>%filter(grepl(str_sub(df_lj$Location.ID, 1, 2), stusps)))['st'],
        #                        str_sub(df_lj$Location.ID, -3, -1),
        #                        sep = ''))
        
        for(i in 1:nrow(df_lj)) {       # for-loop over rows
          df_lj[i, 'fips'] <- strtoi(paste((state_fips_df%>%filter(grepl(str_sub(df_lj[i,'Location.ID'], 1, 2), stusps)))['st'],
                                                                  str_sub(df_lj[i,'Location.ID'], -3, -1),
                                                                  sep = ''))
        }
        
        df_output <- df_lj[,c("fips", "precip_century_annual_mean_1901_2000", "avg_temp_annual_mean_1901_2000", "max_temp_annual_mean_1901_2000", "min_temp_annual_mean_1901_2000")]

        # return dataframe
        write.csv(df_output,".\\get_county_climate.csv", row.names = FALSE)

	    return(df_output)
    }
    else {
        # read in cvs file to append additional columns
        df <- read.csv(".\\get_county_climate.csv", header=TRUE, stringsAsFactors=FALSE)

        return(df)
    }
}