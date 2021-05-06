
######################a.
install.packages("rmarkdown")
#tinytex::install_tinytex()
library(quantmod)
citation("quantmod")
end <- "2020-06-01"
start <- "2010-06-01"
symetf <- c("IEF", "TLT", "SPY", "QQQ")
l <- length(symetf)
w0 <- NULL
for (i in 1:l) {
  dat0 <- getSymbols(symetf[i],
                     src = "yahoo", from = start, to = end, auto.assign = F,
                     warnings = FALSE, symbol.lookup = F
  )
  w1 <- weeklyReturn(dat0)
  w0 <- cbind(w0, w1)
}

dat <- as.matrix(w0)
timee <- as.Date(rownames(dat))
dat <- na.fill(dat, fill = "extend")
colnames(dat) <- symetf
head(dat, 2)
dim(dat)

#Normalize
dat <- scale(dat)

######################b.
pca_dat <- prcomp(dat)
pca_dat$x
summary(pca_dat) #This shows that first two principal components are 

######################c.
# autoencoder in keras
suppressPackageStartupMessages(library(keras))
# set training data
x_train <- as.matrix(dat)

# set model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 4, activation = "linear", input_shape = ncol(x_train)) %>%
  layer_dense(units = 2, activation = "linear", name = "bottleneck") %>%
  layer_dense(units = ncol(x_train),name = "final")
summary(model)

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)

model %>% fit(
  x = x_train, 
  y = x_train, 
  epochs = 100,
  verbose = 0
)

mse.ae2 <- evaluate(model, x_train, x_train)
mse.ae2

#How is this being computed?
#Explain mse

#d)
intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, "final")$output)
intermediate_output <- predict(intermediate_layer_model, x_train)
get_weights(model)
cor(pca_dat$x[,1], intermediate_output[,1])
plot(pca_dat$x[,1], intermediate_output[,1], col=c("red","blue"))

#e)
cor(pca_dat$x[,2], intermediate_output[,2])