#install ggplot dplyr tidyverse
install.packages("ggplot2")
install.packages("dplyr")
install.packages("tidyverse")
install.packages("reshape")
install.packages("randomForest")
install.packages("reshape2")

library(ggplot2)
library(tidyverse)
library(dplyr)


df <- read.csv("placementdata.csv")
head(df)

#check for missing values
sum(is.na(df))


#check for missing values column wise
colSums(is.na(df)) ##found out that no data is missing which is a good news

#check for data types
str(df)

#create a bar plot of placement status
ggplot(df, aes(x = PlacementStatus, fill = PlacementStatus)) +
  geom_bar() +
  scale_fill_manual(values = c("Placed" = "blue", "NotPlaced" = "lightblue")) +
  ggtitle("Placement Status of Students") +
  xlab("Placement Status") +
  ylab("Number of Students") +
  theme_minimal()

#make a scatter plot of aptitude test score vs cgpa data and color by placement status

library(ggplot2)

#create a boxplot of cgpa distribution by placment status
library(ggplot2)

ggplot(df, aes(x = PlacementStatus, y = CGPA, fill = PlacementStatus)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Placed" = "blue", "NotPlaced" = "red")) +
  ggtitle("CGPA Distribution by Placement Status") +
  xlab("Placement Status") +
  ylab("CGPA") +
  theme_minimal() +
  theme(legend.position = "none")

#create a boxplopt of aptitude test score distribution by placement status
ggplot(df, aes(x = PlacementStatus, y = AptitudeTestScore, fill = PlacementStatus)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Placed" = "blue", "NotPlaced" = "red")) +
  ggtitle("Aptitude Test Score Distribution by Placement Status") +
  xlab("Placement Status") +
  ylab("Aptitude Test Score") +
  theme_minimal() +
  theme(legend.position = "none")

ggplot(df, aes(x = PlacementTraining, fill = PlacementStatus)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("Placed" = "blue", "NotPlaced" = "red")) +
  ggtitle("Proportion of Placement Status by Training") +
  xlab("Placement Training") +
  ylab("Proportion") +
  theme_minimal()


#Analysis of correlation across differnet numeric columns
library(reshape2)

# Select relevant numeric columns
numeric_df <- df[, c("CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks", "Internships", "Projects", "Workshops.Certifications")]

# Compute the correlation matrix
cor_matrix <- cor(numeric_df)

# Melt the correlation matrix for ggplot
melted_cor <- melt(cor_matrix)

ggplot(melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, limit = c(-1, 1), name = "Correlation") +
  ggtitle("Correlation Heatmap of Numeric Variables") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


























#analysis of importance of different features using random forest
# Load necessary library
library(randomForest)

# Convert PlacementStatus to a factor (if not already)
# Rename the column for ease of use (if not done already)
names(df)[names(df) == "Workshops/Certifications"] <- "Workshops_Certifications"

# Load required libraries
library(randomForest)
library(ggplot2)

# Build the Random Forest model
set.seed(123)
model_rf <- randomForest(PlacementStatus ~ CGPA + Internships + Projects +
                            + AptitudeTestScore +
                           SoftSkillsRating + SSC_Marks + HSC_Marks,
                         data = df, importance = TRUE)

# Extract variable importance
imp <- importance(model_rf)

# Create a tidy data frame using the "MeanDecreaseGini" metric
imp_df <- data.frame(Feature = rownames(imp),
                     MeanDecreaseGini = imp[, "MeanDecreaseGini"])

# Order features by importance (descending order)
imp_df <- imp_df[order(imp_df$MeanDecreaseGini, decreasing = TRUE), ]

# Plot using ggplot2: color each bar differently by mapping fill to Feature
ggplot(imp_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini, fill = Feature)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Variable Importance from Random Forest",
       x = "Feature",
       y = "Mean Decrease Gini") +
  theme_minimal() +
  theme(legend.position = "none")














str(df)










