#load packages & data
library(e1071)
library(caret)
library(nnet)
library(randomForest)
library(rpart)
library("klaR")

Appointments = read.csv(file.choose())

#look in to the data
dim(Appointments)
str(Appointments)
View(Appointments)


round(prop.table(table(Appointments$No.show))*100)

#No.show and NoNo.show
#Appointments$No.show <- 0
#Appointments$No.show <- ifelse(Appointments$No.show == "No", "Yes","No")
#Appointments$No.show <- NULL

#data Cleaninig
#Split scheduled day by T
Appointments$ScheduledDate <- sub("[T].*", "", Appointments$ScheduledDay)
Appointments$ScheduledTime <- sub(".*[T]", "", Appointments$ScheduledDay)
Appointments$ScheduledDate = as.Date(Appointments$ScheduledDate, format = "%Y-%m-%d")

#Split Appointment day by T
Appointments$AppointmentDate <- sub("[T].*", "", Appointments$AppointmentDay)
Appointments$AppointmentDate = as.Date(Appointments$AppointmentDate, format = "%Y-%m-%d")

#Get weekday from date
Appointments$ScheduledWeekDay <- weekdays(as.Date(Appointments$ScheduledDate))
Appointments$AppointmentWeekDay <- weekdays(as.Date(Appointments$AppointmentDate))
Appointments$AppointmentWeekDay <- as.factor(Appointments$AppointmentWeekDay)

#Calculate the difference between appointment date and scheduled date 
Appointments$DayDifference <- Appointments$AppointmentDate - Appointments$ScheduledDate
Appointments$DayDifference <-  as.numeric(Appointments$DayDifference)

Appointments$Hour <- sapply(strsplit(Appointments$ScheduledTime,":"),`[`, 1)
Appointments$Minute <- sapply(strsplit(Appointments$ScheduledTime,":"),`[`, 2)
Appointments$Second <- sapply(strsplit(Appointments$ScheduledTime,":"),`[`, 3)
Appointments$Second <- sub("[Z].*", "", Appointments$Second)

Appointments$Minute <- as.numeric(Appointments$Minute)
Appointments$Hour <- as.numeric(Appointments$Hour)
Appointments$Second <- as.numeric(Appointments$Second)

Appointments$HourOfDay <- Appointments$Hour+ (Appointments$Minute)/60 + (Appointments$Second)/3600
Appointments$HourOfDay <- round(Appointments$HourOfDay,0)

Appointments$Minute <- NULL
Appointments$Hour <- NULL
Appointments$Second <- NULL
Appointments$ScheduledDay <- NULL
Appointments$AppointmentDay <- NULL
Appointments$ScheduledTime <- NULL
Appointments$ScheduledDate <- NULL
Appointments$AppointmentDate <- NULL

sort(unique(Appointments$DayDifference))
sort(unique(Appointments$Age))
sort(unique(Appointments$HourOfDay))
unique(Appointments$Gender)
unique(Appointments$Scholarship)
unique(Appointments$Hipertension)
unique(Appointments$Diabetes)
unique(Appointments$Alcoholism)
unique(Appointments$Handcap)
unique(Appointments$SMS_received)
unique(Appointments$No.show)
unique(Appointments$ScheduledWeekDay)
unique(Appointments$AppointmentWeekDay)

Appointments <- Appointments[Appointments$DayDifference>=0,]
Appointments <- Appointments[Appointments$Age>=0 & Appointments$Age<=95,]


#Convert to factor
Appointments$Scholarship = as.factor(Appointments$Scholarship)
Appointments$Hipertension = as.factor(Appointments$Hipertension)
Appointments$Diabetes = as.factor(Appointments$Diabetes)
Appointments$Alcoholism = as.factor(Appointments$Alcoholism)
Appointments$Handcap = as.factor(Appointments$Handcap)
Appointments$SMS_received = as.factor(Appointments$SMS_received)

#set seed
set.seed(1234)

#Split data in to 70:30
row_size = floor(0.70 * nrow(Appointments))
train_ind <- sample(seq_len(nrow(Appointments)), size = row_size)
train = Appointments[train_ind,]
test = Appointments[-train_ind,]

round(prop.table(table(train$No.show))*100)
round(prop.table(table(test$No.show))*100)

#remove the unwanted column
#train = train[,colnames(train) %in% c("Gender", "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received", "DayDifference", "AppointmentWeekDay", "ScheduledWeekDay", "ScheduledTime", "No.show")]
#test = test[,colnames(test) %in% c("Gender", "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received", "DayDifference", "AppointmentWeekDay", "ScheduledWeekDay", "ScheduledTime", "No.show")]


#Select the attribute in test for the model
testData = test[,!colnames(test) %in% c("Gender", "PatientId", "AppointmentID", "Neighbourhood", "Hipertension", "ScheduledWeekDay", "No.show")]

y = test[,colnames(test) %in% c("No.show")]

#run the logistic model
modelLogit = glm(No.show ~ Age + Diabetes +Scholarship + Alcoholism + SMS_received + DayDifference + HourOfDay + AppointmentWeekDay, data = train, family = "binomial")

#check the summary of the Logistic model
summary(modelLogit)

#Predict
prediction = predict(modelLogit, testData, type = "response")
prediction
prediction = ifelse(prediction > 0.4, "Yes", "No")
roc.curve(y, Prediction)
confusionMatrix(y,prediction)

#Random Forest
rf <- randomForest(No.show ~ AppointmentWeekDay +DayDifference + SMS_received + Age + HourOfDay +Scholarship, data = train)

#check the summary of the model
summary(rf)

#Predict
prediction = predict(rf, testData)
prediction
confusionMatrix(y,prediction)
roc.curve(y, Prediction)

#Naive Bayes
modelNaive <- naiveBayes(No.show ~ AppointmentWeekDay +DayDifference + SMS_received + Age + HourOfDay +Scholarship, data = train)
summary(modelNaive)
prediction = predict(modelNaive, testData)
unique(prediction)
class(prediction)
table(prediction,y)
confusionMatrix(y,prediction)

#test rpart
modelRPart <- rpart(No.show ~ AppointmentWeekDay +DayDifference + SMS_received + Age + HourOfDay +Scholarship, data = train)
Prediction <- predict(modelRPart, testData, type='class')
accuracy.meas(y,Prediction)
roc.curve(y, Prediction)
confusionMatrix(y,prediction)
head(Prediction)

write.csv(Appointments,file = "Mydata.csv")

#check table
table(train$No.show)


install.packages("ROSE")
library(ROSE)

data_balanced_over <- ovun.sample(No.show ~ AppointmentWeekDay +DayDifference + SMS_received + Age + HourOfDay +Scholarship, data = train, method = "over",N = 123310)$data
table(data_balanced_over$No.show)

data_balanced_under <- ovun.sample(No.show ~ AppointmentWeekDay +DayDifference + SMS_received + Age + HourOfDay +Scholarship, data = train, method = "under",N = 31354,seed=1)$data
table(data_balanced_under$No.show)

data_balanced_both <- ovun.sample(No.show ~ AppointmentWeekDay +DayDifference + SMS_received + Age + HourOfDay +Scholarship, data = train, method = "both", p=0.5,N=77332, seed = 1)$data
table(data_balanced_both$No.show)

data_ROSE <-  ROSE(No.show ~ DayDifference + SMS_received + Age + HourOfDay +Scholarship, data = train, seed = 1)$data
table(data_ROSE$No.show)

tree.rose <-rpart(No.show ~ ., data = data_ROSE)
tree.over <- rpart(No.show ~ ., data = data_balanced_over)
tree.under <- rpart(No.show ~ ., data = data_balanced_under)
tree.both <- rpart(No.show ~ ., data = data_balanced_both)

pred.tree.rose <- predict(tree.rose, newdata = testData)
pred.tree.over <- predict(tree.over, newdata = testData)
pred.tree.under <- predict(tree.under, newdata = testData)
pred.tree.both <- predict(tree.both, newdata = testData)

roc.curve(y, pred.tree.rose[,2])
roc.curve(y, pred.tree.over[,2])
roc.curve(y, pred.tree.under[,2])
roc.curve(y, pred.tree.both[,2])

confusionMatrix(y,pred.tree.rose)
