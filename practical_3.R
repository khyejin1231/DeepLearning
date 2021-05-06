#  mnist using keras ----
library(tidyverse)
library(feather)
pathh_train <- "C:/Users/Paul Stroet/Documents/Deep learning/mnist_train.feather"
pathh_test <- "C:/Users/Paul Stroet/Documents/Deep learning/mnist_test.feather"
trainn <- read_feather(pathh_train)
testt <- read_feather(pathh_test)
trainn <- as.data.frame(trainn)
testt <- as.data.frame(testt)
dim(testt)

# show label ----
Show_label <- function(row, data) {
  tmp <- data.frame(
    x = rep(1:28, times = 28),
    y = rep(28:1, each = 28),
    shade = as.numeric(data[row, -1])
  )
  ggplot(data = tmp) +
    geom_point(aes(x = x, y = y, color = shade), size = 11, shape = 15) +
    theme(
      axis.line = element_blank(),
      axis.text.x = element_blank(), axis.text.y = element_blank(),
      axis.ticks = element_blank(), axis.title.x = element_blank(),
      axis.title.y = element_blank(), legend.position = "none", panel.background = element_blank(),
      panel.border = element_blank(), panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(), plot.background = element_blank()
    ) +
    scale_color_gradient(low = "white", high = "black") +
    geom_text(aes(x = 28, y = 28), label = data[row, 1])
}

ii <- 775
Show_label(ii, as.data.frame(trainn)) # the first column is the label
ii <- 777
Show_label(ii, as.data.frame(trainn)) # the first column is the label
ii <- 778
Show_label(ii, as.data.frame(trainn)) # the first column is the label




# multinomial regression -----------------------------------------------
library(nnet)

scale_norm <- function(x){
  ( x - min(x) ) / ( max(x) - min(x) )
}

Trainx <-  as.data.frame( trainn[,-1] )
Trainx_scaled <- scale_norm(Trainx)
test_scaled <- scale_norm(testt)
Trainy <- as.factor(as.matrix(trainn[,1]))
dat <- data.frame(y= Trainy, as.matrix(Trainx_scaled))

# The following line takes couple of minutes!
multnom <- multinom(y ~., data = dat,  
                    maxit=100, MaxNWts= 7860, 
                    abstol= 10^(-2) )

pred_multnom <- predict(multnom, newdata=test_scaled[,-1], type='probs')
dim(pred_multnom)

pred_multnom <- max.col(pred_multnom) # apply a maximum to determine the class
pred_multnom <- pred_multnom - 1 # to convert to digits

# images incorrectly classified by the multinomial model
which(pred_multnom != testt[,1])
pred_multnom[which(pred_multnom != testt[,1])]

er <- mean(pred_multnom != testt[,1])
print(paste('Accuracy', 1 - er))

library(caret)
library(e1071)
confusion <- caret::confusionMatrix(data= factor(pred_multnom), reference=factor(testt[,1]))
confusion$table

# neural network set up ----
library(reticulate)
reticulate::py_discover_config()
# use_python(python= "")
library(tensorflow)
library(keras)
# tensorflow::tf_config()
tf$constant("Hellow Tensorflow")

# convert, for training in keras
# y_train <- to_categorical(trainn[,1]) %>% as.matrix
# y_test <-  to_categorical(testt[,1]) %>% as.matrix

y_train <- tf$keras$utils$to_categorical(trainn[,1]) %>% as.matrix
y_test  <- tf$keras$utils$to_categorical(testt[,1]) %>% as.matrix

# base model ----------------------------------------------------------
# sequential model
first_model_base <- keras_model_sequential() %>% 
  layer_dense(units = 10, activation = "relu",
              input_shape = c(28*28)) %>% 
  layer_dense(units = 10, activation = "softmax")

# examples of layers
# Create a sigmoid layer:
layer_dense(units = 64, activation ='sigmoid')

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layer_dense(units = 64, kernel_regularizer = regularizer_l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layer_dense(units = 64, bias_regularizer = regularizer_l2(0.01))

# A linear layer with a bias vector initialized to 2.0:
layer_dense(units = 64, bias_initializer = initializer_constant(2.0))


# train and evaluate
first_model_base %>% compile(
  optimizer = optimizer_sgd(), #optimizer_adam, 
  loss = "categorical_crossentropy",
  metrics = list("accuracy", "mse")
)

# summary of the model
first_model_base %>% summary

epochss <- 3 # change to 5 or 10. 
first_model_base <- first_model_base %>% fit(as.matrix(Trainx_scaled), 
                                                     y_train, epochs = epochss, 
                                                     batch_size = 128,
                                             validation_data=list(as.matrix(test_scaled[,-1]),y_test))

# model 1 -------------------------------------------------------------
first_model_drop_out <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu",
              layer_dropout(rate=0.3),
              input_shape = c(28*28)) %>% 
  layer_dense(units = 512, activation = "relu",
              layer_dropout(rate=0.3),
              input_shape = c(28*28)) %>% 
  layer_dense(units = 512, activation = "relu",
              layer_dropout(rate=0.3),
              input_shape = c(28*28)) %>% 
  layer_dense(units = 10, activation = "softmax")

first_model_drop_out %>% compile(
  optimizer = optimizer_sgd(), #optimizer_adam, 
  loss = "categorical_crossentropy",
  metrics = list("accuracy", "mse")
)

first_model_drop_out %>% summary

epochss <- 3 # change to 5 or 10. 
first_model_drop_out <- first_model_drop_out %>% fit(as.matrix(Trainx_scaled), 
                    y_train, epochs = epochss, 
                    batch_size = 128,
                    validation_data=list(as.matrix(test_scaled[,-1]),y_test))

# model 2 ---------------------------------------------------------
second_model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 10, activation = "softmax")

second_model %>% compile(
  optimizer= optimizer_adam(),
  loss = "categorical_crossentropy",
  metrics = c("accuracy", "mse")
)

second_model %>% summary

epochss <- 3 # change to 5 or 10. 
second_model <- second_model %>% fit(as.matrix(Trainx_scaled), 
                    y_train, epochs = epochss, 
                    batch_size = 128,
                    validation_data=list(as.matrix(test_scaled[,-1]),y_test))

# model 3 ----------------------------------------------------------
img_rows <- 28
img_cols <- 28
input_shape <- c(img_rows, img_cols, 1)

mnist <- dataset_mnist(path = "mnist.npz")
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- tf$keras$utils$to_categorical(y_train) %>% as.matrix
y_test  <- tf$keras$utils$to_categorical(y_test) %>% as.matrix

third_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')

third_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c("accuracy","mse")
)

epochss <- 3 # change to 5 or 10. 
third_model <- third_model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = epochss,
  validation_split = 0.2
)

# model_comparison -----------------------------------------------
compare_cx <- data.frame(
  baseline_val = first_model_base$metrics$val_loss,
  first_model_drop_out_val = first_model_drop_out$metrics$val_loss,
  second_model_val = second_model$metrics$val_loss,
  third_model_val = third_model$metrics$val_loss,
  baseline_train = first_model_base$metrics$loss,
  first_model_drop_out_train = first_model_drop_out$metrics$loss,
  second_model_train = second_model$metrics$loss,
  third_model_train = third_model$metrics$loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")

# confusion table 1------------------------------------------------------
second_model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(28*28)) %>% 
  layer_dense(units = 10, activation = "softmax")

second_model %>% compile(
  optimizer= optimizer_adam(),
  loss = "categorical_crossentropy",
  metrics = c("accuracy", "mse")
)

second_model %>% fit(as.matrix(Trainx_scaled), 
                                     y_train, epochs = epochss, 
                                     batch_size = 128,
                                     validation_data=list(as.matrix(test_scaled[,-1]),y_test))

second_model %>% predict(as.matrix(test_scaled[,-1]), batch_size = 128)
pred_second_model <- second_model %>% predict(as.matrix(test_scaled[,-1]), batch_size = 128)

pred_second_model <- max.col(pred_second_model) # apply a maximum to determine the class
pred_second_model <- pred_second_model - 1 # to convert to digits

# images incorrectly classified by the neural network
which(pred_second_model != testt[,1])
pred_second_model[which(pred_second_model != testt[,1])]

# images incorrectly classified by the multinomial model
which(pred_multnom != testt[,1])
pred_multnom[which(pred_multnom != testt[,1])]

# images incorrectly classified by the multinomial model
# and correctly classified by the dl model
track <- NULL
for(i in 1:nrow(testt)){
  track[i] <- 
    ifelse((pred_multnom != testt[,1])[i]&(pred_second_model == testt[,1])[i],
           i,0)
}

# inspection -----
inspection <- data.frame(
  actual <- testt[,1][which(track>0)],
  multnom <- pred_multnom[which(track>0)],
  dl <- pred_second_model[which(track>0)]
)
colnames(inspection) <- c("actual", "multnom", "dl")
# View(inspection)

count <- count(inspection, actual, multnom) %>% arrange(desc(n))
as.data.frame(count)
# head(count)

# confusion table 2 -----
confusion <- caret::confusionMatrix(data= factor(pred_multnom), reference=factor(testt[,1]))
confusion$table
cf_multnom <- confusion$table
confusion <- caret::confusionMatrix(data= factor(pred_second_model), reference=factor(testt[,1]))
confusion$table
cf_first_model <- confusion$table

differences <- matrix(c(0:19),ncol=2)
for(i in 1:10){
  differences[i,2] <-
    cf_first_model[i,i] - cf_multnom[i,i]
}
colnames(differences) <- c("digit","difference")
View(differences)
