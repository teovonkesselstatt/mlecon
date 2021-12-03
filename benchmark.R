################################################################################
#################     Benchmark: Regresion Logistica
################################################################################

set.seed(123); id = sample(nrow(datos),0.8*nrow(datos))

Train=datos[id,] # Conjunto de training
Test=datos[-id,]  # Conjunto de test


# Recordemos que la libreria glmnet usa matrices, no data frames
matrix_train = model.matrix( ~ .-1, Train)
matrix_test = model.matrix(~.-1, Test)

dim(matrix_train)
dim(matrix_test)

# x_train <- as.matrix(matrix_train[,-70])
x_train <- matrix_train[,-70]
dim(x_train)
y_train = as.matrix(matrix_train[,70]) # 0/1 flag.
dim(y_train)

x_test <- as.matrix(matrix_test[,-70])
y_test<- as.matrix(matrix_test[,70])

######################################################
######### REGRESION LOGISTICA  #######################
######################################################

logit.reg = glmnet(x = x_train, 
                   y = y_train,
                   family = 'binomial', # problema de clasificación
                   alpha = 1 , 
                   lambda = 0, # sin regularizacion
                   standardize = TRUE)

# Ver resumen del modelo:
summary(logit.reg$beta)

# Tabla de frecuencias (churn y no-churn)
freq_table <- table(y_train) / dim(y_train)[1] 
prob_priori <- as.numeric(freq_table[2])
freq_table
prob_priori # Probabilidad de que un dato de Train sea Revenue:  15.53%


pred_tr = predict(logit.reg, s = 0 , newx = x_train, type = 'response')


####################### Performance en el TRAIN: ###############################

# El umbral de corte que vamos a usar para decidir sobre una nueva observacion
# si prob>prob_priori --> clasificamos como 1

y_hat = as.numeric(pred_tr >= prob_priori) 


# Matriz de Confusion
matriz.confusion = table(y_hat, y_train)
matriz.confusion

tasa_error_tr = 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)

######################################################
tasa_error_tr # 16.28% USANDO PROB PRIORI EN TRAIN ###
######################################################

pred_train <- prediction(pred_tr, y_train)
perf_train <- performance(pred_train,"tpr","fpr")

# Graficamos la curva ROC
plot(perf_train, main="Curva ROC en el conjunto de TRAIN", colorize=T)

auc <- performance(pred_train, measure = "auc")
auc@y.values[[1]] # 0.9055

####################### Performance en el TEST: ################################


# Predicciones sobre datos TEST 
pred_test = predict(logit.reg, s = 0 , newx = x_test, type = 'response')

# Clasificamos en [0,1] en funcion de una probababilidad a priori
y_hat = as.numeric(pred_test >= prob_priori) 

# Matriz de Confusion
matriz.confusion = table(y_hat, y_test)
matriz.confusion

# Matriz expresada en prob.
# matriz.confusion/dim(y_test)[1]*100

# Tasa de error 
tasa_error <- 1-sum(diag(matriz.confusion))/sum(matriz.confusion)

##################################################
tasa_error # 18.65% USANDO PROB PRIORI EN TEST ###
##################################################

pred2 <- prediction(pred_test, y_test)
perf2 <- performance(pred2,"tpr","fpr")

# Graficamos la curva ROC
plot(perf2, main="Curva ROC en el conjunto de TEST", colorize=T)

auc_test <- performance(pred2, measure = "auc")
auc_test@y.values[[1]] # 0.8745


######################## ELIGIENDO EL THRESHOLD ################################

# Hago Validation Set Approach

set.seed(1); val_id = sample(nrow(matrix_train),0.3*nrow(matrix_train))

matrix_val = matrix_train[val_id,]
matrix_train2 = matrix_train[-val_id,]

x_val <- as.matrix(matrix_val[,-70])
y_val <- as.matrix(matrix_val[,70])

# Predicciones sobre datos de VALIDACION
pred_val = predict(logit.reg, s = 0 , newx = x_val, type = 'response')

thresh = seq(0.05,0.5,length.out = 200)
thresh

# Optimizamos el threshold eligiendo el que maximiza F1.
F2.est = c();
beta = 2

for(i in 1:length(thresh)){
  y_hat = as.numeric(pred_val >= thresh[i])
  TP = sum( y_hat*y_val==1 )     # True +
  FP = sum( y_hat*(1-y_val)==1 ) # False +
  FN = sum( (1-y_hat)*y_val==1 ) # False -
  prec = TP/(TP + FP) ; rec = TP/(TP + FN)
  F2.est[i] = (1+beta^2)*(prec*rec)/((beta^2)*prec + rec)
}

F2.est 
plot(thresh, F2.est, type = 'b', bty = 'n')
thresh.opt = thresh[which(F2.est==max(F2.est))]
abline(v = thresh.opt , col = 'red', lty = 2 )

# Diferencia de Punto de Corte Optimo vs Prob a Priori
thresh.opt;  prob_priori

# Performance en Test:
pred_test = predict(logit.reg, s = 0 , newx = x_test, type = 'response')
y_hat = as.numeric(pred_test>= thresh.opt)

TP = sum( y_hat*y_test==1 )     # True +
FP = sum( y_hat*(1-y_test)==1 ) # False +
FN = sum( (1-y_hat)*y_test==1 ) # False -
prec = TP/(TP + FP) ; rec = TP/(TP + FN)

F2.est = (1+beta^2)*(prec*rec)/((beta^2)*prec + rec)

matriz.confusion = table(y_hat, y_test)
matriz.confusion

tasa_error_test = 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error_test # 18.32% USANDO THRESH.OPT. 
F2.est # 64.90 DE F2 score!














































################################################################################
########################### USANDO REWEIGHTING #################################
################################################################################

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
gamma =  seq(0.6,0.9,length.out = 100)
F2.gamma = c() 

for(j in 1:length(gamma)){
  w = ifelse(Train$Revenue == 1, 1+gamma[j], 1-gamma[j]) #Le damos los pesos a las observaciones
  logit.reg = glmnet(x = x_train,y = y_train, weights = w,
                          family = 'binomial', alpha = 1,lambda = 0)
  pred_val = predict(logit.reg, s = 0 , newx = x_val, type = 'response')
  thresh = seq(0.05,0.5,length.out = 200)

  # Optimizamos el threshold eligiendo el que maximiza F1.
  F2.est.2 = c();
  beta = 2
  
  for(i in 1:length(thresh)){
    y_hat = as.numeric(pred_val >= thresh[i])
    TP = sum( y_hat*y_val==1 )     # True +
    FP = sum( y_hat*(1-y_val)==1 ) # False +
    FN = sum( (1-y_hat)*y_val==1 ) # False -
    prec = TP/(TP + FP) ; rec = TP/(TP + FN)
    F2.est.2[i] = (1+beta^2)*(prec*rec)/((beta^2)*prec + rec)
  }
  F2.gamma[j] <- max(F2.est.2)
  print(j)
}

print(F2.gamma)

plot(gamma, F2.gamma, type = 'b')
abline(v = gamma[which.max(F2.gamma)] , col = 'red', lty = 2 )
best.gamma = gamma[which.max(F2.gamma)] # gamma = 0.74
best.gamma


max(F2.gamma) # score que nos da el reweighting
max(F2.est) # score que nos da el treshold optimization


### Elijo el mejor:

w = ifelse(Train$Revenue == 1, 1+best.gamma, 1-best.gamma) #Le damos el peso gamma = 0.8

#Entrenamos el modelo con el valor de lambda optimizado y con los pesos w:
logit.reg = glmnet(x = x_train,y = y_train, weights = w,
                        family = 'binomial', alpha = 1,lambda = 0)

# Optimizamos el umbral (nu) con el F4 para cada valor de w:
pred = predict(logit.reg, s = 0, newx = x_val, type = 'response')

thresh = seq(0.1,0.5,length.out = 100)
F4.est = c()

for(i in 1:length(thresh)){
  y_hat = as.numeric(pred >= thresh[i])
  TP = sum( y_hat== 1  & y_val==1 )    # True Positives/ 1 y 'yes'
  FP = sum( y_hat == 1 & y_val==1 )     # False Positives/ 1 y 'no'
  FN = sum( y_hat == 0 & y_val==1 )    # False Negatives / 0 y 'yes'
  TN = sum( y_hat == 0 & y_val==1 )     # True  Negatives / 0 y 'no'
  prec = TP/(TP + FP) ; rec = TP/(TP + FN)
  F4.est[i] = (1+beta^2)*(prec*rec)/(beta^2*prec + rec)*100
}

thresh.opt = thresh[which(F4.est==max(F4.est))] #elegimos el umbral óptimo

# Una vez que tenemos el umbral óptimo y el lambda óptimo para cada w, calculamos 
# el F1:
pred_test = predict(logit.reg, s = 0, newx = x_test, type = 'response')

y_hat = as.numeric(pred_test >= thresh.opt)

pred_test <- prediction(y_hat, y_test)
perf_test <- performance(pred_test,"tpr","fpr")

# Graficamos la curva ROC
plot(perf_test, main="Curva ROC en el conjunto de TEST", colorize=T)

auc_test <- performance(pred_test, measure = "auc")
auc_test@y.values[[1]] # 0.8745, DE NUEVO, NO LE IMPORTA EL THRESHOLD








