rm(list=ls())
dev.off()

### Cargamos las librerías: 
require(glmnet)|| install.packages('glmnet')  #regularización de modelos lineales
require(ROCR)  || install.packages('ROCR')
require(corrplot) || install.packages('corrplot')  
require(ggplot2) || install.packages('ggplot2')


library(glmnet)
library(ROCR)
library(corrplot)
library(ggplot2)

######################
###      DATOS     ###
######################

# Link repositorio: https://www.kaggle.com/blastchar/telco-customer-churn
setwd('/Users/delfinaricordi/Documents/2021/Segundo Semestre/ML Prácticas/Clase 5')

# Cargamos los datos
churn <- read.csv('Telco_churn.csv')
churn$X <- NULL
churn$state <- NULL

### TRANSFORMAMOS LOS TIPOS DE LAS VARIABLES:

# Hay variables 'chr' que deberían ser categorías 
for (i in 1:19){
  if (typeof(churn[,i])=="character"){
    churn[,i]=factor(churn[,i])}
}

# Si estuvieramos haciendo un análisis completo:
# Ingenieria de features, corrplots, outliers (clase pasada)


##### DESBALANCE DE LA MUESTRA #####
table(churn$churn)

ggplot(data=churn, aes(churn))+
  geom_bar(aes(fill=churn), position="dodge")



# Hay muchos mas 'no' que 'yes'. 
# El modelo de la semana pasada fallaba para predecir churn=1
# Eso puede ser por el gran desbalance de la base de datos. 
# EJEMPLO TRIVIAL: Si para este caso no entrenamos ningún modelo 
# y predecimos que ninguno va a hacer churn vamos a tener una accuracy
# cercana al 85% (si el promedio de ocurrencias en la muestra es representativo
# de lo que ocurrre en la población.)

# Más en:
# https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28


### ESTRATEGIAS PARA RESOLVER CLASES DESBALANCEADAS: 
# 1. Optimizar el threshold
# 2. Rebalancear la muestra: Reweighting 
# 3. Minimizar costos/ maximizar beneficios


##########################################################
############     TRAIN VALIDATION Y  TEST      ###########
##########################################################

set.seed(123); id = sample(nrow(churn),0.8*nrow(churn))

Train=churn[id,] # Conjunto de training
Test=churn[-id,]  # Conjunto de test

dim(Train)
head(Train)
dim(Test)

# Vamos a hacer Validatoin Set CV, entonces separamos TRAIN en 2 grupos: 
set.seed(1234); id_validation = sample(nrow(Train),0.7*nrow(Train))

Train=Train[id_validation,] 
Validation = Train[-id_validation,]


# Recordemos que la libreria glmnet usa matrices, no data frames
matrix_train = model.matrix( ~ .-1, Train)
matrix_test = model.matrix(~.-1, Test)
matrix_val = model.matrix(~.-1, Validation)

dim(matrix_train)
dim(matrix_test)

x_train <- as.matrix(matrix_train[,-21])
y_train = as.matrix(matrix_train[,21]) # 0/1 flag.

x_test <- as.matrix(matrix_test[,-21])
y_test<- as.matrix(matrix_test[,21])

x_val <- as.matrix(matrix_val[,-21])
y_val <- as.matrix(matrix_val[,21])

dim(x_train)
head(x_test)
head(x_train,2)


#############################
# 1. Optimizar el threshold #
#############################

#### Como elegimos el umbral de corte para clasificar 0/1 a clientes de test?
#### Optimizando el threshold para maximizar F1: https://en.wikipedia.org/wiki/F1_score


# The F_{beta} score is the harmonic mean of the precision and recall: 
#FORMULA: (1+beta^2)*(prec*rec)/(beta^2*prec + rec)*100

# Precision: True Positives / (True Positives + False Positives)
#           cuantos clasificados positivos fueron correctos 

# Recall: True Positives / (True Positives + False Negatives)
#         A que proporcion de los Positivos le pegamos 

#F2 le da dos veces mas ponderacion a los falsos negativos que a los falsos positivos



#Modelo simple de Logit sin regularizacion
lasso.sin.reg = glmnet(x = x_train , 
                       y = y_train,
                       family = 'binomial', # problema de clasificación
                       alpha = 1 , 
                       lambda = 0, # sin regularizacion
                       standardize = TRUE)
# help(glmnet)
summary(lasso.sin.reg$beta)

# Predicciones sobre datos de VALIDACION
pred_tr = predict(lasso.sin.reg, s = 0 , newx = x_val, type = 'response')


# Antes: tabla de frecuencias (churn y no-churn)
freq_table <- table(y_train) / dim(y_train)[1] 
prob_priori <- as.numeric(freq_table[1])
freq_table
prob_priori


thresh = seq(0.01,0.7,length.out = 100)
thresh


# Optimizamos el threshold eligiendo el que maximiza F1.
F2.est = c();
for(i in 1:length(thresh)){
  y_hat = as.numeric(pred_tr >= thresh[i])
  TP = sum( y_hat*y_val==1 )     # True +
  FP = sum( y_hat*(1-y_val)==1 ) # False +
  FN = sum( (1-y_hat)*y_val==1 ) # False -
  prec = TP/(TP + FP) ; rec = TP/(TP + FN)
  F2.est[i] = (1+2^2)*(prec*rec)/(2^2*prec + rec)*100
}
F2.est 
plot(thresh, F2.est, type = 'b', bty = 'n')
thresh.opt = thresh[which(F2.est==max(F2.est))]
abline(v = thresh.opt , col = 'red', lty = 2 )

# Diferencia de Punto de Corte Optimo vs Prob a Priori
thresh.opt;  prob_priori

# Performance en Test:
pred_test = predict(lasso.sin.reg, s = 0 , newx = x_test, type = 'response')

y_hat = as.numeric(pred_test>= thresh.opt)
matriz.confusion = table(y_hat, y_test)
matriz.confusion # Caen los Falsos Positivos 


# Curva ROC y AUC
dev.off();
pred2 <- prediction(pred_test, y_test)
perf <- performance(pred2,"tpr","fpr")

# Graficamos la curva ROC
plot(perf, main="Curva ROC", colorize=T)

####################
#  2. REWEIGHTING  #
####################

# Nuevo Hiperparámetro: gamma
# gamma en (0,1) es un hiperparámetro con el que
# controlo cuanto over/under-sampling hago de cada clase. 
# Mide el peso que le damos a las aboservaciones al momento de entrenar el modelo.  
# EJEMPLO: gamma=0.1 
#         Le damos peso 1,1 a las obs churn=yes, y peso 0.9 a las churn=no


# Ajustando lambda vía 5-fold cross-validation y gamma con validation set approach.  
gamma = c(0,0.01,0.1,0.2,0.3,0.5, 0.6, 0.7,0.75, 0.8,0.85, 0.9) 
F2.gamma = c() 
bestlam <- c()
for(j in 1:length(gamma)){
  # 1.1 Optimizamos lambda (regulariza el modelo)
  w = ifelse(Train$churn=='yes',1+gamma[j],1-gamma[j]) #Le damos los pesos a las observaciones
  set.seed(1)
  grid.l = exp(-10:10)
  cv.out = cv.glmnet(x_train, y_train, 
                     family = 'binomial', # Modelo logístico.
                     weights = w,
                     type.measure="auc",  # Métrica de CV.
                     lambda = grid.l, 
                     alpha = 1, # LASSO.
                     nfolds = 5)
  bestlam[j] = cv.out$lambda.min
  
  #Entrenamos el modelo con el valor de lambda optimizado y con los pesos w:
  logistic.lasso = glmnet(x = x_train,y = y_train, weights = w,
                          family = 'binomial', alpha = 1,lambda = bestlam[j])
  
  # Optimizamos el umbral (nu) con el F4 para cada valor de w:
  pred = predict(logistic.lasso, s = bestlam[j] , newx = x_val, type = 'response')
  
  thresh = seq(0.001,0.6,length.out = 100)
  F4.est = c();
  for(i in 1:length(thresh)){
    y_hat = as.numeric(pred >= thresh[i])
    TP = sum( y_hat== 1  & y_val==1 )    # True Positives/ 1 y 'yes'
    FP = sum( y_hat == 1 & y_val==1 )     # False Positives/ 1 y 'no'
    FN = sum( y_hat == 0 & y_val==1 )    # False Negatives / 0 y 'yes'
    TN = sum( y_hat == 0 & y_val==1 )     # True  Negatives / 0 y 'no'
    prec = TP/(TP + FP) ; rec = TP/(TP + FN)
    F4.est[i] = (1+4^2)*(prec*rec)/(4^2*prec + rec)*100
  }
  thresh.opt = thresh[which(F4.est==max(F4.est))] #elegimos el umbral óptimo
  
  # Una vez que tenemos el umbral óptimo y el lambda óptimo para cada w, calculamos 
  # el F1:
  pred.val = predict(logistic.lasso, s = bestlam[j] , newx = x_val, type = 'response')
  y_hat = as.numeric(pred >= thresh[i])
  TP = sum( y_hat== 1  & y_val==1)    # True Positives/ 1 y 'yes'
  FP = sum( y_hat ==1 & y_val==0 )     # False Positives/ 1 y 'no'
  FN = sum( y_hat == 0 & y_val==1)     # False Negatives / 0 y 'yes'
  TN = sum( y_hat == 0 & y_val==0)      # True  Negatives / 0 y 'no'
  prec = TP/(TP + FP) ; rec = TP/(TP + FN)
  F2.gamma[j] <- (1+2^2)*(prec*rec)/(2^2*prec + rec)*100
  print(j)
}

print(F4.est)
print(F2.gamma)

plot(gamma, F2.gamma, type = 'b')
abline(v = gamma[which.max(F2.gamma)] , col = 'red', lty = 2 )
#gamma = 0.8












