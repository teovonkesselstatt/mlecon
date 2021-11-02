### Regresión logística ###
rm(list=ls())
dev.off()

### Cargamos las librerías: 
require(glmnet)|| install.packages('glmnet')  #regularización de modelos lineales
require(ROCR)  || install.packages('ROCR')
require(ggplot2) || install.packages('ggplot2')


library(glmnet)
library(ROCR)
library(ggplot2)


######################
###      DATOS     ###
######################

# Link repositorio: https://www.kaggle.com/blastchar/telco-customer-churn
setwd('/Users/delfinaricordi/Documents/2021/Segundo Semestre/ML Prácticas/Clase 5')

# Cargamos los datos
churn <- read.csv('Telco_churn.csv')
churn$X <- NULL

# Churn rate: porcentaje de clientes que dejan de usar el producto de una compañía

# Objetivo: predecir si un cliente va a seguir comprando o va a hacer churn


######################################################
################## Analisis Exploratorio #############
######################################################

### MISSINGS: 
sum(is.na(churn)==TRUE)  #No hay missings 

## Si los hubiera: Vemos cómo se reparten en las variables
sapply(churn, function(x) sum(is.na(x)))



### TRANSFORMAMOS LOS TIPOS DE LAS VARIABLES:

# Hay variables 'chr' que deberían ser categorías 

# typeof() nos dice de qué tipo es la variable. 
# en vez de hacer el cambio de las variables 'chr' a factor una por una 
# podemos hacer un loop para que lo haga por nosotros 

for (i in 1:20){
  if (typeof(churn[,i])=="character"){
    churn[,i]=factor(churn[,i])}
}

# Vemos cuántas observaciones de cada categoría hay:
table(churn$churn)
### Hay un desbalance en la muestra. (Más de cómo resolver este problema la clase que viene)

print(names(churn))
head(churn, 3)
str(churn)


### ANALISIS EXLORATORIO GRÁFICO: 

### VARIABLES CATEGÓRICAS
g1 <- ggplot(data=churn, aes(state))+
  geom_bar(aes(fill=churn), position="fill") # podemos graficar por categorias!

g2 <- ggplot(data=churn, aes(area_code))+
  geom_bar(aes(fill=churn), position="fill")

g3 <- ggplot(data=churn, aes(international_plan))+
  geom_bar(aes(fill=churn), position="fill")

g4 <- ggplot(data=churn, aes(voice_mail_plan))+
  geom_bar(aes(fill=churn), position="fill")


# VARIABLES NUMÉRICAS
g5 <- ggplot(data=churn, aes(account_length, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g5
g6 <- ggplot(data=churn, aes(number_vmail_messages, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g6
g7 <- ggplot(data=churn, aes(total_day_minutes, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g7
g8 <- ggplot(data=churn, aes(total_day_calls, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g8
g9 <- ggplot(data=churn, aes(total_day_charge, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g9
g10 <- ggplot(data=churn, aes(total_eve_minutes, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g10
g11 <- ggplot(data=churn, aes(total_eve_calls, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g11
g12 <- ggplot(data=churn, aes(total_eve_charge, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g12
g13 <- ggplot(data=churn, aes(total_night_calls, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g13
g14 <- ggplot(data=churn, aes(total_night_charge, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g14
g15 <- ggplot(data=churn, aes(total_intl_minutes, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g15
g16 <- ggplot(data=churn, aes(total_intl_calls, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g16
g17 <- ggplot(data=churn, aes(total_intl_charge, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g17
g18 <- ggplot(data=churn, aes(number_customer_service_calls, color = churn, fill = churn))+
  geom_density(alpha = 0.1)
g18
# ORDENARLOS EN EL MISMO ESPACIO:
require(gridExtra) || install.packages('gridExtra')
library(gridExtra)

grid.arrange(g1, g2, g3, g4,  ncol=2)
grid.arrange(g5, g6, g7, g8, ncol=2)
grid.arrange(g9, g10, g11, g12, ncol=2)


## CORRELACION ENTRE VARIABLES (Graficos lindos de correlaciones)
require(corrplot) || install.packages('corrplot')  
library(corrplot)

numeric.var <- sapply(churn, is.numeric)
corr.matrix <- cor(churn[,numeric.var])
corrplot(corr.matrix, main="\nCorrelation Plot for Numerical Variables", method="number")

#las variables 'charge' agregan la misma informacion que minutes. 
# Eliminamos las minute:
churn$total_day_minutes <- NULL 
churn$total_eve_minutes <- NULL 
churn$total_night_minutes <- NULL
churn$total_intl_minutes <- NULL



numeric.var <- sapply(churn, is.numeric)
corr.matrix <- cor(churn[,numeric.var])
corrplot(corr.matrix, main="\nCorrelation Plot for Numerical Variables", method="number")
#Notar: las correlaciones entre ellas son iguales a cero. Pero 
# lo que nos interesa es con churn: 
churn$churn_numeric <- as.numeric(churn$churn)


corr.matrix_1 <- cor(churn[,c(17, 2, 6, 7, 8, 9, 10)])
corrplot(corr.matrix_1, main="\nCorrelation Plot for Numerical Variables", method="number")

corr.matrix_2 <- cor(churn[,c(17,11, 12, 13, 14, 15)])
corrplot(corr.matrix_2, main="\nCorrelation Plot for Numerical Variables", method="number")

churn$churn_numeric <- NULL

## Chequamos los outliers: 
boxplot(churn$account_length)
boxplot(churn$number_vmail_messages)
boxplot(churn$total_day_calls)
#repetir para el resto


is.numeric(churn[,10])==TRUE

for(i in 1:16){
  if(is.numeric(churn[,i])==TRUE){
    q0.01 = quantile(churn[,i], 0.01)  
    q0.99 = quantile(churn[,i], 0.99) 
    churn[which(churn[,i]<q0.01),i] = q0.01
    churn[which(churn[,i]>q0.99),i] = q0.99
}}



##########################################################
##################      TRAIN Y TEST      ################
##########################################################

set.seed(123); id = sample(nrow(churn),0.7*nrow(churn))

Train=churn[id,] # Conjunto de training
Test=churn[-id,]  # Conjunto de test

dim(Train)
head(Train)
dim(Test)

# Recordemos que la libreria glmnet usa matrices, no data frames
matrix_train = model.matrix( ~ .-1, Train)
matrix_test = model.matrix(~.-1, Test)

dim(matrix_train)
dim(matrix_test)

x_train <- as.matrix(matrix_train[,-67])
dim(x_train)
y_train = as.matrix(matrix_train[,67]) # 0/1 flag.
dim(y_train)

x_test <- as.matrix(matrix_test[,-67])
y_test<- as.matrix(matrix_test[,67])


dim(x_train)
head(x_test)
head(x_train,2)


######################################################
######### REGRESION LOGISTICA  #######################
######################################################

# Entrenamos modelo de regresion logística
# log(P(Y=1|X)/P(Y=0|X))=b0 + b1x1 + ... + bp xp
# Predicciones con el modelo: P(Y=1|X)=e^(b0 + b1x1 + ... + bp xp)/(1+e^(b0 + b1x1 + ... + bp xp))


help(glm)
rlogit.churn = glm(churn ~.,  data = Train, family = binomial(link = "logit"))

# Ver resumen del modelo:
summary(rlogit.churn)

# Resaltar variables importantes:
#     international_planyes
#     voice_mail_planyes             
#     total_intl_charge
#     total_intl_calls
#     number_customer_service_calls

########################################################
######### REGRESION LOGISTICA SIN REGULARIZACION #######
########################################################

# Recordatorio: 
# Ridge -> alpha = 0
# Lasso -> alpha = 1
# ENets -> alpha in (0,1)

# Entrenamos el modelo Regresion Logistica (LASSO) - Sin Regularizacion
lasso.sin.reg = glmnet(x = x_train , 
                       y = y_train,
                       family = 'binomial', # problema de clasificación
                       alpha = 1 , 
                       lambda = 0, # sin regularizacion
                       standardize = TRUE)
# help(glmnet)
summary(lasso.sin.reg$beta)

# Predicciones sobre datos TRAIN 
pred_tr = predict(lasso.sin.reg, s = 0 , newx = x_train, type = 'response')

# Predicciones sobre datos TEST 
pred = predict(lasso.sin.reg, s = 0 , newx = x_test, type = 'response')
head(pred) # s = 0 (sin regularizacion)

# Evaluando PERFORMANCE del modelo
# Matriz de confusion
# Tasa de acierto
# Curva ROC y AUC
# Matriz de confusion sobre el conjunto de TEST

# Que hace 'table()'?
help(table)  #Cuenta la combinacion de los niveles de factores

# Tabla de frecuencias (churn y no-churn)
freq_table <- table(y_train) / dim(y_train)[1] 
prob_priori <- as.numeric(freq_table[1])
freq_table

# El umbral de corte que vamos a usar para decidir sobre una nueva observacion
# si prob>prob_priori --> clasificamos como 1
prob_priori

######################
# Performance en TRAIN
######################

# Clasificamos en [0,1] en funcion de una probababilidad a priori
y_hat = as.numeric(pred_tr >= prob_priori) 
head(y_hat)

# Matriz de Confusion
matriz.confusion = table(y_hat, y_train)
matriz.confusion

# Matriz expresada en proporciones
matriz.confusion/dim(y_train)[1]*100

# Tasa de error 
# estimacion puntual con datos de train
tasa_error_tr <- 1-sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error_tr # 13%
# Es conveniente evaluar el modelo con la tasa de acierto unicamente?

##### Curva ROC y Area baja la curva ROC (AUC)
### Primero obtenemos las propabilidades a posteriori, luego:
dev.off();
pred2 <- prediction(pred_tr, y_train)
perf <- performance(pred2,"tpr","fpr")

# Que hace 'performance'?
help(performance)

# Graficamos la curva ROC
plot(perf, main="Curva ROC", colorize=T)

auc <- performance(pred2, measure = "auc")
auc@y.values[[1]] # 0.83


######################
# Performance en TEST
######################

# Clasificamos en [0,1] en funcion de una probababilidad a priori
y_hat = as.numeric(pred >= prob_priori) 

# Matriz de Confusion
matriz.confusion = table(y_hat, y_test)
matriz.confusion

# Matriz expresada en prob.
matriz.confusion/dim(y_test)[1]*100

# Tasa de error 
tasa_error <- 1-sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error # ~13%

##### Curva ROC y Area baja la curva ROC (AUC)
#dev.off();
pred2 <- prediction(pred, y_test)
perf <- performance(pred2,"tpr","fpr")

# Graficamos la curva ROC
plot(perf, main="Curva ROC", colorize=T)

auc <- performance(pred2, measure = "auc")
auc@y.values[[1]] # 0.83(aprox)

# Reportamos las metricas obtenidas
m_reglog<- c(tasa_error, auc@y.values[[1]], te_train_test)
m_reglog

# Comparacion de metricas TRAIN vs. TEST 
tasa_error_tr/tasa_error

######################################################
######### REGRESION LOGISTICA + REGULARIZACION #######
######################################################

# Creamos una secuencia de valores para lambda
grid.l =exp(seq(10, -10 , length = 100))
plot(grid.l)
# Entrenamos el modelo Regresion Logistica - Regularizacion
lasso.reg = glmnet(x = x_train , 
                   y = y_train,
                   family = 'binomial', 
                   alpha = 1 , 
                   lambda = grid.l, 
                   standardize = TRUE)


# Visualuzamos la evolucion de los valores de las variables 
# en funcion de los distintos log(lambda)
plot(lasso.reg, xvar = "lambda", label = TRUE)


# A medida que lambda aumenta, mayor penalidad, menos cantidad de features

# Predicciones sobre datos TEST 
# elijo arbitrariamente un valor de lambda (s)
pred = predict(lasso.reg, s = 0.008 , newx = x_test, type = 'response')
head(pred) # s = 0 (con regularizacion)


###############################################################
###     Ajustando lambda via 5-fold cross-validation.      ####  
###############################################################
set.seed(123)
cv.out = cv.glmnet(x_train, y_train, 
                   family = 'binomial', # Modelo logistico.
                   type.measure="auc", # Metrica del CV (no depende del umbral).
                   lambda = grid.l, 
                   alpha = 1, # LASSO.
                   nfolds = 5)

cv.out
plot (cv.out)
# Graficamos como varia AUC en funcion del valor de lambda 
# Recordamos que significan las lineas verticales
# the left vertical line in our plot shows us where the CV-error curve hits its minimum. 
# The right vertical line shows us the most regularized model with CV-error 
#           within 1 standard deviation of the minimum. 
# Mas info sobre paquete glmnet: 
# https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#log

# Seleccionamos el mejor lambda (aquel que minimiza el CV-error)
bestlam = cv.out$lambda.min
bestlam # poca regularizacion

## Vemos la lista de los coeficientes que no son cero
rownames(coef(lasso.reg, s = bestlam))[coef(lasso.reg, s = bestlam)[,1]!= 0]

## Para ver los valores: 
coef(lasso.reg, s = bestlam)[coef(lasso.reg, s = bestlam)[,1]!= 0]


# Predicciones sobre el conjunto de TEST con el mejor lambda
pred = predict(lasso.reg, s = bestlam , newx = x_test, type = 'response')

y_hat = as.numeric(pred>= prob_priori)

# Evaluamos Performance en TEST
# Matriz de confusion 
matriz.confusion = table(y_hat, y_test)
matriz.confusion

# Tasa de Error
tasa_error <- 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error # 13%

# Curva ROC y AUC
pred2 <- prediction(pred, y_test)
perf <- performance(pred2,"tpr","fpr")
auc <- performance(pred2,"auc")
auc@y.values[[1]] # 0.846 ()

# Reportamos las metricas obtenidas
m_reglogRcv<- c(tasa_error, auc@y.values[[1]], te_train_test)
m_reglogRcv
m_reglog

#Train vs test
tasa_error_tr/tasa_error

# Obs: Los modelos tienen una performance muy parecida con y sin regularizacion



#####################################################
############ CLASIFICACION CON K VECINOS ############
#####################################################

##### RECORDAR: con K vecinos hay que escalar las variables ANTES

x_train_scaled <- as.data.frame(scale(x_train))
dim(x_train_scaled)
x_test_scaled <- as.data.frame(scale(x_test))

library(FNN)

y_knn <- knn(train = x_train_scaled, 
             test=x_test_scaled,
             cl = y_train, 
             prob=T,
             k = 3)

length(y_knn) #realiza las predicciones sobre Test

#########################
#########   VC  #########
#########################

#No podemos elegir los parametros sobre TEST (out of sample)

#Podemos hacer Validation set approach y separar TRAIN en 2 grupos: 
set.seed(1234); id_validation = sample(nrow(x_train_scaled),0.7*nrow(x_train_scaled))

x_knn_train =x_train_scaled[id_validation,] # Conjunto de training
x_knn_val =x_train_scaled[-id_validation,] # Conjunto de validation

y_knn_train = y_train[id_validation,]
y_knn_train <- factor(y_knn_train)
y_knn_val = y_train[-id_validation,]
dim(y_knn_train)
dim(y_knn_val)
###### Generamos la grilla de valores de k
valores.k = c(5,10,20,30, 40,50, 60, 70, 80, 90, 100) 

## Tenemos que decidir qué métrica de error usar
# La funcion de la curva ROC no funciona para knn, asi que usamos tasa de error: 

tasa_knn <- c() # Vamos a ir llenando la tasa para cada valor de k


for(i in 1:length(valores.k)){ # i recorre los valores del parámetro k.
  y_knn <- knn(train = x_knn_train,     #Corremos la regresion sobre Train
               test=x_knn_val,          # Hacemos la prediccion sobre validation
               cl = y_knn_train, 
               prob=T,
               k = valores.k[i])        #Para cada valor de la grilla
  # Matriz de confusion 
  matriz.confusion = table(y_knn, y_knn_val)
  matriz.confusion
  
  # Tasa de Error
  tasa_knn[i] <- 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
  print(i)  
}


# Plot:
plot(valores.k,tasa_knn,type='b',pch=20, main='Validation set VC') # :)

which.min(tasa_knn)
tasa_knn[which.min(tasa_knn)]  # tasa de error ~15% 
valores.k[which.min(tasa_knn)] # k = 20




#######################
# Performance en TEST #
#######################

#Primero corremos el modelo con el K=20

y_knn <- knn(train = x_train_scaled, 
             test=x_test_scaled,
             cl = y_train, 
             prob=T,
             k = 20)

length(y_knn)

# Evaluamos Performance en TEST
# Matriz de confusion 
matriz.confusion = table(y_knn, y_test)
matriz.confusion

# Tasa de Error
tasa_error <- 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error # 13,73%

###PROBLEMA: no predice ningún churn=1. Ese problema lo vamos a resolver la clase que viene. 






