######CLEAR Global Environment ######
rm(list = ls())

#######verify that all R packages needed for this project are installed, and activate libraries ######
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
library(readr)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(tidyverse)
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(knitr)
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
library(ggplot2)
if (!require(scales)) install.packages('scales')
library(scales)


###### Read in Data to R ######

#Read in hotel rds file, it has been saved in github ripository: https://github.com/tclex7/havard_edx_final_project/blob/master/hotel.rds
hotel_reviews <- read_rds("hotel.rds")

#remove duplicate rows in the dataframe
hotel_reviews <- distinct(hotel_reviews)

###### Overview of Data ######
glimpse(hotel_reviews)
summary(hotel_reviews)
head(hotel_reviews)

#select only the variables that we will be using for machine learning algorithm 
hotel_reviews <- hotel_reviews %>% select(name, reviews.rating,reviews.username,province)
#rename variables 
colnames(hotel_reviews) <- c("hotel","rating","user","state")

#check to see if any of the variables have N/As
sapply(hotel_reviews,function(x)sum(is.na(x)))

#Since there is only one N/A for user, we will call that user "Mr. Unknown"
hotel_reviews[is.na(hotel_reviews)] <- "Mr. Unknown"

#now we can verify no N/As exist in the dataset
sapply(hotel_reviews,function(x)sum(is.na(x)))

#Summary of distinct reviews, hotels, users, and states
hotel_reviews %>% summarize(total_reviews = n(),
                            total_hotels = n_distinct(hotel),
                            total_users = n_distinct(user),
                            total_states = n_distinct(state))

#check how many different ratings were given
n_distinct(hotel_reviews$rating)

#We see 1 is the smallest rating and 5 was the highest
summary(hotel_reviews$rating)

#table breakdown of all ratings
data.frame(table(hotel_reviews$rating))

###### Visaulize data ######

#Histrogram showing distribution of rating, we can see that majority of ratings were either 5s or 4s, 
#make sure to zoom in to get a better view of the histogram
hotel_reviews %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = .25, color = "blue") +
  scale_x_continuous(breaks=seq(1, 5, .25)) +
  scale_y_continuous(labels=comma) +
  labs(x="hotel Rating", y="# of Ratings") +
  ggtitle("Hotel Ratings")

#number of ratings per user, we can see that the majority of users only submitted one review
hotel_reviews %>% 
  group_by(user) %>%
  summarize(total_user = as.numeric(n())) %>%
  ggplot(aes(total_user)) +
  geom_histogram(bins = 50) +
  scale_x_log10()

#we will find out exactly what portion of the users only completed 1 review in the dataset
hotel <- hotel_reviews %>% 
  group_by(user) %>%
  summarize(total_user = as.numeric(n())) %>% 
  arrange(desc(total_user)) %>%
  mutate(one_or_more = ifelse(total_user==1,"just_one","more_than_one"))
table(hotel$one_or_more)

#looks like about 92% of users only completed 1 review
round(prop.table(table(hotel$one_or_more))*100,0)


#Visualize top 20 state by rating distribution, we can see California and Florida get the most reviews
hotel_reviews %>% 
  group_by(state) %>%
  summarize(total_state = as.numeric(n())) %>%
  arrange(desc(total_state)) %>%
  slice(1:20) %>%
  ggplot(aes(x = reorder(state, -total_state),total_state), colour ="blue") +
  geom_col() +
  scale_y_continuous(labels=comma) +
  labs(x="State", y="# of Ratings") +
  ggtitle("Total Ratings by State")

###### partition out data ###### 
#if you have R version 3.5 or below use set.seed(1), below you can see what your current version is
set.seed(1, sample.kind="Rounding")
version$version.string

test_index <- createDataPartition(y = hotel_reviews$rating, times = 1, p = 0.4, list = FALSE)
hotel_train <- hotel_reviews[-test_index,]
temp <- hotel_reviews[test_index,]

# Make sure user and hotel in hotel_test set are also in hotel_train set

hotel_test <- temp %>% 
  semi_join(hotel_train, by = "hotel") %>%
  semi_join(hotel_train, by = "user")

# Add rows removed from hotel_test set back into hotel_train set
removed <- anti_join(temp, hotel_test)
hotel_train <- rbind(hotel_train, removed)

#remove variables no longer needed
rm(test_index, temp, removed)

#rename variables to make it easier to run functions
train_set <- hotel_train
test_set <- hotel_test
rm(hotel_train, hotel_test)

###### Modeling ###### 


#Define RMSE function, this will measure how far our predictions are from true rating
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#For this model we will use a simple average of the training set for prediction
mu <- mean(train_set$rating)
mu

#RCompute RMSE on the test set
average_rmse <- RMSE(test_set$rating, mu)
average_rmse

#Show RMSE in a clean way using knitr
rmse_results <- data_frame(method = "Average Hotel Rating Model", RMSE = average_rmse)
rmse_results %>% knitr::kable()

##Now we will add the hotel effect to the model
hotel_avgs <- train_set %>% 
  group_by(hotel) %>% 
  summarize(b_i = mean(rating - mu))

#visualize how close ratings are to the mean
hotel_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("blue"))

#We will calculate predicted ratings
predicted_ratings <- mu + test_set %>% 
  left_join(hotel_avgs, by='hotel') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="hotel Effect Model",
                                     RMSE = model_1_rmse ))

rmse_results %>% knitr::kable()

# Now we will add users to the previous model
train_set %>% 
  group_by(user) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 10, color = "red")


user_avgs <- train_set %>% 
  left_join(hotel_avgs, by='hotel') %>%
  group_by(user) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(hotel_avgs, by='hotel') %>%
  left_join(user_avgs, by='user') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Hotel + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

##We will add state to the previous model, and follow similar process as previous lines of code
train_set %>% 
  group_by(state) %>% 
  summarize(b_s = mean(rating)) %>% 
  ggplot(aes(b_s)) + 
  geom_histogram(bins = 10, color = "green")

state_avgs <- train_set %>% 
  left_join(hotel_avgs, by='hotel') %>%
  left_join(user_avgs, by='user') %>%
  group_by(state) %>%
  summarize(b_s = mean(rating - mu - b_i - b_u))

predicted_ratings <- test_set %>% 
  left_join(hotel_avgs, by='hotel') %>%
  left_join(user_avgs, by='user') %>%
  left_join(state_avgs, by='state') %>%
  mutate(pred = mu + b_i + b_u + b_s) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Hotel + User + State Effects Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

#Now we will try using regularization on Hotel to improve RMSE score
lambdas <- seq(0, 10, 0.1)
mu <- mean(train_set$rating)
just_the_sum <- train_set %>% 
  group_by(hotel) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='hotel') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
lambda
mu <- mean(train_set$rating)
hotel_reg_avgs <- train_set %>% 
  group_by(hotel) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

data_frame(original = hotel_avgs$b_i, 
           regularlized = hotel_reg_avgs$b_i, 
           n = hotel_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)


predicted_ratings <- test_set %>% 
  left_join(hotel_reg_avgs, by='hotel') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized hotel Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

#Now we will try using regularization on user to improve RMSE score
lambdas <- seq(0, 10, 0.1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(hotel) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="hotel") %>%
    group_by(user) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "hotel") %>%
    left_join(b_u, by = "user") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized hotel + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

#Now we will try using regularization on state to improve RMSE score
lambdas <- seq(0, 10, 0.1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(hotel) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="hotel") %>%
    group_by(user) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_s <- train_set %>%
    left_join(b_i, by="hotel") %>%
    left_join(b_u, by="user") %>%
    group_by(state) %>%
    summarize(b_s = sum(rating - b_i - b_u - mu)/(n()+l))
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "hotel") %>%
    left_join(b_u, by = "user") %>%
    left_join(b_s, by = "state") %>%
    mutate(pred = mu + b_i + b_u + b_s) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized hotel + User Effect Model+ State",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

##Hotel and State regularization effect without user, since user was negatively affected our score
lambdas <- seq(0, 10, 0.1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(hotel) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_s <- train_set %>% 
    left_join(b_i, by="hotel") %>%
    group_by(state) %>%
    summarize(b_s = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "hotel") %>%
    left_join(b_s, by = "state") %>%
    mutate(pred = mu + b_i + b_s) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized hotel + State Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()



