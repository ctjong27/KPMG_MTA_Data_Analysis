# https://www.epa.gov/system/files/documents/2022-05/ctyfactbook2021.xlsx

library(tidyverse)
library(readxl)

get_county_pollution <- function() {
    if(!file.exists("get_county_pollution.csv")) {
        
        temp.file <- paste(tempfile(),".xlsx",sep = "")
        download.file("https://www.epa.gov/system/files/documents/2022-05/ctyfactbook2021.xlsx", temp.file, mode = "wb")

        # skip countes number of lines to skip in excel file to find column head names
        tmp <- read_excel(temp.file, skip = 2)
        df <- data.frame(tmp)

        names(df)

        # keep only the state, county, fips, area_land, area_water
        df_output <- df %>% select(c("County.FIPS.Code", "O3............8.hr..ppm."))
        colnames(df_output)  <- c("fips", "pollutant_o3")
        # df_output[,"poverty_percentage"] <- round(df_output[,"poverty_percentage"], 1)


        print(names(df_output))

        print(head(df_output))


        # return dataframe
        write.csv(df_output,".\\get_county_pollution.csv", row.names = FALSE)

	    return(df_output)
    }
    else {
        # read in cvs file to append additional columns
        df <- read.csv(".\\get_county_pollution.csv", header=TRUE, stringsAsFactors=FALSE)

        return(df)
    }
}