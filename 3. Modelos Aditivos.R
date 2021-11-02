rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica

library("readxl")
setwd("C:/Users/teovo/Google Drive/DI TELLA/8vo semestre/Machine Learning/data")
datos <- read_excel('CIT_2019_Cambridge.xlsx', col_names = TRUE )


attach(datos)
library(ggplot2)
library(dplyr)

# X = islamic margin of victory 
# Y = female high school percentage 


datos_2 <- datos %>% filter(X > -20 & X<20) 

treat <- datos_2  %>% filter(T ==1)
control <- datos_2  %>% filter(T == 0)


datos_2$T <- as.factor(datos_2$T)

ggplot(data = datos_2, aes(x = X,  y = Y, color = T)) +
  geom_point(alpha = .1) +
  geom_smooth(method='lm',se=F)

ggplot(data = datos_2, aes(x = X,  y = Y, color = T)) +
  geom_point(alpha = .1) +
  geom_smooth(se=F)


#### Fitteamos regresiones lineales locales ###
library(psych)
library(splines)

# Creamos las matrices X e Y 
X <- as.numeric(as.matrix(datos_2$X))
Y <- as.numeric(as.matrix(datos_2$Y))
Tr <- as.numeric(as.matrix(datos_2$T))

data <- as.data.frame(cbind(X, Y, Tr))

### ESTIMO EL LADO DEL CONTROL ###

### Separo en train y test: 
### Primero para el control: 
set.seed(1)
id.train_c <- sample(nrow(control), 0.75*nrow(control)) 
train_X_control <- as.matrix(control[id.train_c,1])
train_Y_control <- as.matrix(control[id.train_c,2])

test_X_control <- as.matrix(control[-id.train_c,1])
test_Y_control <- as.matrix(control[-id.train_c,2])



# 1) Regresión lineal: 
reglin_control <- lm(Y ~ X, subset = train_X_control, data = control)
summary(reglin_control)

pred_reglin <- predict(reglin_control, newdata = as.data.frame(test_X_control))

ECM_reglin = sum( (pred_reglin - test_Y_control)^2 )/ nrow(test_Y_control) #  ECM = 90,49

#Gráfico
data_frame <- data.frame(cbind(train_X_control, train_Y_control))
ggplot(data_frame, aes(X, Y) ) +
  geom_point() +
  geom_smooth(method='lm', col='red', se=F)



# 2) Polinomio
pol = lm(Y ~ poly(X,3,raw=TRUE), data=control, subset=id.train_c )
summary(pol)
pred_pol = predict(pol, newdata = as.data.frame(test_X_control))
ECM_pol =  sum( (pred_pol - test_Y_control)^2 )/ nrow(test_Y_control) #90,62

#Validación cruzada sobre el grado del polinomio: 
d.cv = c(2:6); ecm.pol = c()

for(i in 1:length(d.cv)){
  pol = lm(Y ~ poly(X,i,raw=TRUE), data=control, subset=id.train_c )
  pred_pol = predict(pol, newdata = as.data.frame(test_X_control))
  ecm.pol[i] =  sum( (pred_pol - test_Y_control)^2 )/ nrow(test_Y_control)
}
ecm.pol
which.min(ecm.pol) # la posición 5, polinomio de grado 6 
ECM_bestpol = ecm.pol[which.min(ecm.pol)] # ECM=88,47

# Fitteamos para el valor óptimo obtenido: 
best_pol = lm(Y ~ poly(X,6,raw=TRUE), data=control, subset=id.train_c )
summary(best_pol)

# Gráfico: 
data_frame <- data.frame(cbind(train_X_control, train_Y_control))
ggplot(data_frame, aes(X, Y) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ poly(x,6,raw=TRUE), se=F)+
  geom_smooth(method='lm', col='red', se=F)



# 3) Splines:
require(splines) || install.packages('splines'); library(splines)

?bs # Similar al comando "poly
nodos = quantile(train_X_control, probs = seq(0, 1, 1/3))  # Divide la muestra en 3, según cuantiles
head(bs(train_X_control, knots = nodos ,degree = 3)) # bs contruye la matriz "X" del modelo de splines.


spline = lm(Y ~ bs(X, knots = nodos ,degree = 3), data=control, subset=id.train_c) 
summary(spline)
spline_pred <- predict(spline, newdata=data.frame(test_X_control))

sum( (spline_pred - test_Y_control)^2 )/ nrow(test_X_control) #  ECM = 88,49


# Gráfico: 
data_frame <- data.frame(cbind(train_X_control, train_Y_control))
ggplot(data_frame, aes(X, Y) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ bs(x, knots = nodos ,degree = 3), se=F, col='green')+
  stat_smooth(method = lm, formula = y ~ poly(x,6,raw=TRUE), se=F, col='blue')+
  geom_smooth(method='lm', col='red', se=F)


######## Validación Cruzada ##########


head(train_X_control)
head(bs(x = as.matrix(train_X_control), 
   df = 2, # Toma los valores del vector "k" 
   intercept = F))


nodos = quantile(train_X_control, probs = seq(0, 1, 1/3))

numero = 1/c(2:7)
ecm.val = c()

spline = lm(Y ~ bs(X, knots = quantile(train_X_control, probs = seq(0, 1, 0.5)) ,degree = 3), data=control, subset=id.train_c) 
summary(spline)
spline_pred <- predict(spline, newdata=data.frame(test_X_control))


for(i in 1:length(numero)){ #recorre los valores de la grilla de numero
  spline = lm(Y ~ bs(X, knots = quantile(train_X_control, probs = seq(0, 1, numero[i])) ,degree = 3), data=control, subset=id.train_c) 
  summary(spline)
  spline_pred <- predict(spline, newdata=data.frame(test_X_control))
  
  ecm.val[i]=sum( (spline_pred - test_Y_control)^2 )/ nrow(test_X_control)
}

ecm.val; length(ecm.val)

which.min(ecm.val) # posición 2:
numero[2]          # dividimos la muestra en 3

ecm.val[which.min(ecm.val) ]  #ECM=88.49


# Gráfico: 
data_frame <- data.frame(cbind(train_X_control, train_Y_control))
ggplot(data_frame, aes(X, Y) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ bs(x, knots = nodos ,degree = 3), se=F, col='green')+
  stat_smooth(method = lm, formula = y ~ poly(x,6,raw=TRUE), se=F, col='blue')+
  geom_smooth(method='lm', col='red', se=F)




### ESTIMO EL LADO DEL TRATAMIENTO ###

### Separo en train y test: 
### Primero para el control: 
set.seed(1)
id.train_t <- sample(nrow(treat), 0.75*nrow(treat)) 
train_X_treat <- as.matrix(treat[id.train_t,1])
train_Y_treat <- as.matrix(treat[id.train_t,2])

test_X_treat <- as.matrix(treat[-id.train_t,1])
test_Y_treat <- as.matrix(treat[-id.train_t,2])


#Regresión lineal: 
reglin_treat <- lm(Y ~ X, subset = train_X_treat, data = treat)
summary(reglin_treat)

pred_reglin_treat <- predict(reglin_treat, newdata = as.data.frame(test_X_treat))

sum( (pred_reglin_treat - test_Y_treat)^2 )/ nrow(test_Y_treat) #  ECM = 91,10

reglin_treat <- lm(Y ~ X, subset = train_X_treat, data = treat)
summary(reglin_treat)

pred_reglin_treat <- predict(reglin_treat, newdata = as.data.frame(test_X_treat))

ECM_reglin_T = sum( (pred_reglin_treat - test_Y_treat)^2 )/ nrow(test_Y_treat) #  ECM = 91,10

# Grafico
data_frame_t <- data.frame(cbind(train_X_treat, train_Y_treat))
ggplot(data_frame_t, aes(X, Y) ) +
  geom_point() +
  geom_smooth(method='lm', col='red', se=F)



# 2) Polinomio
pol_treat = lm(Y ~ poly(X,3,raw=TRUE), data=treat, subset=id.train_t )
summary(pol)
pred_pol_treat = predict(pol_treat, newdata = as.data.frame(test_X_treat))
ECM_pol_treat =  sum( (pred_pol_treat - test_Y_treat)^2 )/ nrow(test_Y_treat) #90,62

#Validación cruzada sobre el grado del polinomio: 
d.cv = c(2:6); ecm.pol = c()

for(i in 1:length(d.cv)){
  pol = lm(Y ~ poly(X,i,raw=TRUE), data=treat, subset=id.train_t )
  pred_pol = predict(pol, newdata = as.data.frame(test_X_treat))
  ecm.pol[i] =  sum( (pred_pol - test_Y_treat)^2 )/ nrow(test_Y_treat)
}
ecm.pol
which.min(ecm.pol) # la posición 1, polinomio de grado 2
ECM_bestpol_T = ecm.pol[which.min(ecm.pol)] # ECM=87,99

# Fitteamos para el valor óptimo obtenido: 
best_pol = lm(Y ~ poly(X,2,raw=TRUE), data=treat, subset=id.train_t )
summary(best_pol)

# Gráfico: 
data_frame_t <- data.frame(cbind(train_X_treat, train_Y_treat))
ggplot(data_frame_t, aes(X, Y) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ poly(x,2,raw=TRUE), se=F)+
  geom_smooth(method='lm', col='red', se=F)



# 3) Splines:
require(splines) || install.packages('splines'); library(splines)

?bs # Similar al comando "poly
nodos = quantile(train_X_treat, probs = seq(0, 1, 1/3))  # Divide la muestra en 3, según cuantiles
head(bs(train_X_treat, knots = nodos ,degree = 2)) # bs contruye la matriz "X" del modelo de splines.


spline_t = lm(Y ~ bs(X, knots = nodos ,degree = 2), data=treat, subset=id.train_t) 
summary(spline_t)
spline_pred_t <- predict(spline_t, newdata=data.frame(test_X_treat))

sum( (spline_pred_t - test_Y_treat)^2 )/ nrow(test_X_treat) #  ECM = 91,99


# Gráfico: 
data_frame <- data.frame(cbind(train_X_treat, train_Y_treat))
ggplot(data_frame, aes(X, Y) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ bs(x, knots = nodos ,degree = 2), se=F, col='green')+
  stat_smooth(method = lm, formula = y ~ poly(x,2,raw=TRUE), se=F, col='blue')+
  geom_smooth(method='lm', col='red', se=F)


######## Validación Cruzada ##########

nodos = quantile(train_X_control, probs = seq(0, 1, 1/3))

numero = 1/c(1:7)
ecm.val = c()

for(i in 1:length(numero)){ #recorre los valores de la grilla de numero
  spline = lm(Y ~ bs(X, knots = quantile(train_X_treat, probs = seq(0, 1, numero[i])) ,degree = 3), data=treat, subset=id.train_t) 
  summary(spline)
  spline_pred <- predict(spline, newdata=data.frame(test_X_treat))
  
  ecm.val[i]=sum( (spline_pred - test_Y_treat)^2 )/ nrow(test_X_treat)
}

ecm.val; length(ecm.val)

which.min(ecm.val) # posición 1:
numero[1]          # no dividir

ecm.val[which.min(ecm.val) ]  #ECM=90,94


# Gráfico de todos: 
data_frame_t <- data.frame(cbind(train_X_treat, train_Y_treat))
ggplot(data_frame_t, aes(X, Y) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ bs(x, knots = quantile(train_X_treat, probs = seq(0, 1,1)) ,degree = 3), se=F, col='green')+
  stat_smooth(method = lm, formula = y ~ poly(x,6,raw=TRUE), se=F, col='blue')+
  geom_smooth(method=lm, col='red', se=F)



#Tratamiento y control:

ggplot(data = datos_2, aes(x = X,  y = Y, col=T)) +
  geom_point(alpha = .1) +
  stat_smooth(method = lm, formula = y ~ bs(x, knots = quantile(datos_2$X, probs = seq(0, 1,1)) ,degree = 3), se=F)+
  stat_smooth(method = lm, formula = y ~ poly(x,6,raw=TRUE), se=F)+
  geom_smooth(method=lm, se=F)









