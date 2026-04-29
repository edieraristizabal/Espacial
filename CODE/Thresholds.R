# title: "Channelization Threshold"
# author: "Maria Isabel Arango"
# date: "2026-04-27"


## 1.2 Import rasters and create the basic pixel by pixel dataframes

#Import DEM, Slope, Flow accumulation layers, stack them and create a
#dataframe with pixel information 

library(terra)
library(sf)
library(data.table)

# Load shapefile with AOI 
area_shape <- st_read(ruta_shapefile/con_AOI.shp)

# Resolution for flow accumulation
facc_res <- 12.5 #Alos Palsar

# Import rasters
uridem <- terra::rast(ruta_DEM.tif)
urislope <- terra::rast(ruta_Slope_degrees.tif)
urifacc <- terra::rast(ruta_flow_accumulation_cells)

# Reproject shapefile geometry to match the CRS of the raster
area_shape <- st_transform(area_shape, crs(uridem))

#Crop rasters to AOI
uridem <- crop(uridem, area_shape) %>% mask(area_shape)
urislope <- crop(urislope, area_shape) %>% mask(area_shape)
urifacc <- crop(urifacc, area_shape) %>% mask(area_shape)

# Convert rasters to data.table
df <- as.data.table(as.data.frame(c(uridem, urislope, urifacc), xy = TRUE))

# Rename columns
setnames(df, 
         old = names(df)[3:5],  # Assuming raster data starts at the 3rd column
         new = c("DEM", "Slope_degrees", "Facc"))  # Standardized names

# Clean and organize slope
df[Slope_degrees < 1, Slope_degrees := 1] #Slopes of 0 give error
df[, Slope_gradient := tan(Slope_degrees * (pi / 180))]

#Logscale gradient
df[, Slope_gradient_Log:= log10(Slope_gradient)]

# Clean and organize flow accumulation
df[Facc <= 0, Facc := 1] #Values of 0 give error
df[, Drain_Area := (abs(Facc) * (facc_res ^ 2)) / 1e6]

#Logscale drain area
df[, Drain_Area_Log:= log10(Drain_Area)]

# Replace NA values with 0
for (col in names(df)) {
  set(df, which(is.na(df[[col]])), col, 0)
}

# Standardize variables
df[, Stan_Slope_gradient_Log := as.numeric(scale(df$Slope_gradient_Log))]
df[, Stan_Drain_Area_Log := as.numeric(scale(df$Drain_Area_Log))]

write.csv(df, file = ruta_df.csv)
  
# Output of this chunk:
# Columns on each file: 
# - "x" 
# - "y" 
# - "DEM" 
# - "Slope_degrees
# - "Facc" 
# - Slope_gradient
# - Slope_gradient_log
# - Drain_Area (Km2)
# - Drain Area Log
# - Stan_Slope_gradient_Log
# - Stan_Drain_Area_Log

#Compute threshold

library(mcp)
library(codatools)
library(readr)
library(ggplot2)

# Open each study area file, select 10000 points, run the mpc model and
# plot slope-area curves with the 90th percentile as the channelizaion
# threshold.

# Filter rows where your_column_name > 0
filtered_df <- df[df$DEM > 0, ]
    
# Sample from the filtered dataframe
n <- 10000 
df_sample <- filtered_df[sample(1:nrow(filtered_df), 
                                min(n, nrow(filtered_df))), ]

# Define the model
mcp_model <- list(
  Stan_Slope_gradient_Log ~ 1,
  ~ 0 + Stan_Drain_Area_Log
)

# Fit the model
fit <- mcp(mcp_model, data = df_sample)

# Save the fitted model object
result_file <- gsub(".csv", "_fit.rds", basename(mcp_resultados))
result_path <- file.path(ruta_a_resultado, result_file)
saveRDS(fit, result_path)

# Save the posteriors
post1 <- coda_df(fit$mcmc_post[1])
post2 <- coda_df(fit$mcmc_post[2])
post3 <- coda_df(fit$mcmc_post[3])
post_all <- rbind(post1, post2, post3)

# Calculate threshold in cells (90th quantile) and area
ecdf_fun <- ecdf(post_all$cp_1)
quantile_90 <- as.numeric(quantile(ecdf_fun, probs = 0.90))

mean_area <- mean(df$Drain_Area_Log)
sd_area <- sd(df$Drain_Area_Log)

facc_res <- 12.5
thres_area <- 10^((quantile_90 * sd_area) + mean_area)
thres_cells <- (1000000 * thres_area) / (facc_res^2)

threshold_data <- data.frame(Quant_90_area_log_stan = quantile_90,
                             Quant_90_area = thres_area,
                             Threshold_cells = thres_cells)
#Threshold cells es el limite en celdas


# Write thethresholds dataframe to file
write.csv(threshold_data, file = thresholds_file, row.names = FALSE)


# Write the posteriors to file
post_file <- gsub(".csv", "_post.csv", basename(posteriores))
write.csv(post_all, file = paste0(ruta_a_posteriores, post_file), row.names = FALSE)

# Save the plot
plot_file <- gsub(".csv", "_fit_plot.png", basename(csv_file))
plot <- plot(fit, geom_data = 'point', cp_dens = TRUE, q_fit = TRUE, q_predict = c(0.1, 0.9)) + 
  theme_bw(15) +
  labs(x = expression("Catchment Area (km"^{2}*")"), y = "Slope", size = 16) +
  geom_vline(xintercept = quantile_90, color = "orange", size = 1.1) +
  scale_y_continuous(labels = function(x) parse(text = paste("10^", x, sep = ""))) +
  scale_x_continuous(labels = function(x) parse(text = paste("10^", x, sep = ""))) +
  theme(axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16))

ggplot2::ggsave(filename = plot_file, plot, path = plots_path, width = 20, height = 12, units = "cm")



#The outputs of this chunck: 
#-mcp_models: the output of the models (.rds files) 
#- mcp plots: the output plots with the threshold line drawn in orange 
#- mcp_psoteriors: Posteriors of each stuy area 
#-mcp_thresholds: one .csv file with the quantil 90 of the posteriors, its value in area (km2) and the value in cells (12.5\*12.5m)
