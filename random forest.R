##########################################################
################     RANDOM FOREST     ###################
##########################################################

rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica

datos = read.table('datos.csv',
                   sep=',',header=T,dec='.',na.strings = "N/A")

quiero.ver.con.weights = FALSE

if (!quiero.ver.con.weights){
  datos$weights = 1
}

datos$weights[which.min(datos$weights)]    

datos$Month <- factor(datos$Month, levels = c("Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), ordered = TRUE)
datos$OperatingSystems <- factor(datos$OperatingSystems)
datos$Browser <- factor(datos$Browser)
datos$Region <- factor(datos$Region)
datos$TrafficType <- factor(datos$TrafficType)
datos$VisitorType <- factor(datos$VisitorType)
datos$Revenue <- factor(datos$Revenue)
datos$Weekend <- factor(datos$Weekend)

######### Separo en TRAIN, TEST Y VALIDATION ###################################


library(rpart); library(ROCR); library(rpart.plot); library(ggplot2); library(randomForest)

set.seed(123); id = sample(nrow(datos),0.8*nrow(datos))

Train=datos[id,] # Conjunto de training
Test=datos[-id,]  # Conjunto de test

# Vamos a hacer Validatoin Set CV, entonces separamos TRAIN en 2 grupos: 
set.seed(123); id_validation = sample(nrow(Train),0.3*nrow(Train))

Validation = Train[id_validation,]
Train=Train[-id_validation,] 

###########################
###    Random Forest    ###
###########################

valores.m = c(8,9,10,12,14,16)      # Posibles valores a considerar de "m"
valores.maxnode = c(80,100,120,140,160)   # Posibles valores  de complejidad de c/ árbol en el bosque.
parametros = expand.grid(valores.m = valores.m,valores.maxnode = valores.maxnode, auc = 0) 
head(parametros,3) # Matriz de 30 filas x 3 columnas.
set.seed(123)
for(i in 1:dim(parametros)[1]){ # i recorre la grilla de parámetros.
  forest.oob  = randomForest(Revenue~. - weights,
                             data=Train,
                             mtry = parametros[i,1],   # Chupo "m" de la primera columna de la matriz parámetros. 
                             ntree=100,                # B (en la práctica B debería ser suficientemente grande como para permitirle al modelo profitar de la reducción de la varianza al promediar modelos)
                             maxnodes = parametros[i,2], # Chupo "complexity parameter" de la segunda columna de la matriz parámetros. 
                             nodesize = 50,
                             sampsize = nrow(Train)*0.6,
                             importance = F,
                             weights = weights)  
  pred.acp = predict(forest.oob,type='prob', newdata = Validation)
  pred2 <- prediction(pred.acp[,2], Validation$Revenue)
  auc_test <- performance(pred2, measure = "auc")
  
  
  parametros[i,3] = auc_test@y.values[[1]] 
  print(i)                                                                                   
}

library("scatterplot3d");

parametros[which.max(parametros[,3]),]

set.seed(123)
modelo.final = randomForest(Revenue~. - weights,                   
                            data=Train,
                            mtry = parametros[which.max(parametros[,3]),1], 
                            ntree = 100,               
                            maxnodes = parametros[which.max(parametros[,3]),2],
                            nodesize = 50,
                            sampsize = nrow(Train)*0.8,
                            importance=T)

pred.acp.Test = predict(modelo.final, type='prob', newdata = Test)

pred2 <- prediction(pred.acp.Test[,2], Test$Revenue)
perf2 <- performance(pred2,"tpr","fpr")

plot(perf2, main="Curva ROC (in-sample) Random Forest", colorize=T)

auc_test <- performance(pred2, measure = "auc")
auc_test@y.values[[1]] # 0.8795, bien


###################### EXTENSIONES: ############################################

# 1. Optimizar el threshold: 

# Para que devuelva la probabilidad:
pred.acp.Prob= predict(modelo.final, type='prob', newdata = Validation)


#Optimizamos el Threshold: 
thresh = seq(0.01,0.8,length.out = 100)


# Optimizamos el threshold eligiendo el que maximiza F1
F2.est = c();
beta = 2
for(i in 1:length(thresh)){
  #Clasificamos a las observaciones
  y_hat = as.numeric(pred.acp.Prob[,2] >= thresh[i])
  TP = sum( y_hat== 1  & Validation$Revenue=="1" )    # True  +/1/'yes'
  FP = sum( y_hat == 1 & Validation$Revenue=="0" )     # False +/1/'yes'
  FN = sum( y_hat == 0 & Validation$Revenue=="1" )    # False -/0/'no'
  TN = sum( y_hat == 0 & Validation$Revenue=="0" )     # True  -/0/'no'
  prec = TP/(TP + FP) ; rec = TP/(TP + FN)
  F2.est[i] <- (1+beta^2)*(prec*rec)/(beta^2*prec + rec)*100
}

thresh.opt = thresh[max(which(F2.est==max(F2.est)))]

pred.acp.Prob.test= predict(modelo.final,type='prob', newdata = Test)

beta = 2

#Clasificamos a las observaciones
y_hat = as.numeric(pred.acp.Prob.test[,2] >= thresh.opt)
TP = sum( y_hat== 1  & Test$Revenue=="1" )    # True  +/1/'yes'
FP = sum( y_hat == 1 & Test$Revenue=="0" )     # False +/1/'yes'
FN = sum( y_hat == 0 & Test$Revenue=="1" )    # False -/0/'no'
TN = sum( y_hat == 0 & Test$Revenue=="0" )     # True  -/0/'no'
prec = TP/(TP + FP) ; rec = TP/(TP + FN)
F2.est <- (1+beta^2)*(prec*rec)/(beta^2*prec + rec)*100

matriz.confusion = table(y_hat, Test$Revenue)
matriz.confusion

tasa_error_test = 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error_test # 13.87%. AHORA: IGUAL
F2.est #71.25. AHORA: IGUAL
auc_test@y.values[[1]] # 0.8918. AHORA: IGUAL


x11(); scatterplot3d(parametros,type = "h", color = "blue", main = "AUC para cada par de hiperparámetros")





