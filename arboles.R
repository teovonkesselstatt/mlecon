##########################################################
############     TRAIN, VALIDATION Y  TEST     ###########
##########################################################

library(rpart); library(ROCR); library(rpart.plot); library(ggplot2)

set.seed(123); id = sample(nrow(datos),0.8*nrow(datos))

Train=datos[id,] # Conjunto de training
Test=datos[-id,]  # Conjunto de test

dim(Train)
dim(Test)

# Vamos a hacer Validatoin Set CV, entonces separamos TRAIN en 2 grupos: 
set.seed(123); id_validation = sample(nrow(Train),0.3*nrow(Train))

Validation = Train[id_validation,]
Train=Train[-id_validation,] 

######################
###     Árboles    ###
######################

# Repasamos los parametros del modelo rpart
help(rpart) #method: class y anova para regresión.
#weights: igual que antes, podemos repesar a las observaciones

# xval: folds para CV



set.seed(123) 
arbol = rpart(Revenue ~ ., data=Train, 
                    method="class", 
                    parms = list( split = "gini"),       # Métrica con la que determinan los cortes.
                    control= rpart.control(minsplit = 100,       # Cantidad minima de observaciones en nodo (antes de partir)
                                           xval = 10,            # cantidad de folds de validación
                                           cp = 0.0001,           # Umbral de mejora mínima (equivale a "alpha" en escala [0,1]).
                                           maxdepth = 10 )      # Longitud maxima del arbol.
)

rpart.plot(arbol, type = 2, cex=0.75)

# Importancia de cada variable en la decisión:
arbol$variable.importance # Importancia de cada covariable (feature).

importancia <- arbol$variable.importance
nombres <- names(importancia)

df <- cbind(importancia, nombres)
zeros <- cbind(rep(0,16), nombres)
df<- data.frame(rbind(df, zeros))

df$importancia <- as.numeric(df$importancia)
df$nombres <- as.factor(df$nombres)

ggplot(df, aes(x=importancia, y=nombres))+geom_line(aes(colour=nombres, size=2), show.legend = F)



# Estimación por VC del tamaño optimo del árbol:
### Como "podar el árbol":
errores <-printcp(arbol) # Elegimos el valor de cp que minimiza el error 
which.min(errores[,4])
errores[which.min(errores[,4]),1]
# estimado por VC (columna xerror).


arbol.podado = prune(arbol,cp = errores[which.min(errores[,4]),1] )

rpart.plot(arbol.podado, type =2, cex=0.75) 


### Predicciones con el arbol:

pred= predict(arbol.podado, type='class')
head(pred,3)


# Matriz de Confusion (datos de train):
table(pred,Train$Revenue)

1-sum(diag(table(pred,Train$Revenue)))/sum(table(pred,Train$Revenue))
#T.Error = 0.0899

# Curva ROC
library(ROCR)

# Curva ROC
pred.acp = predict(arbol.podado,type='prob', newdata = Train)

pred2 <- prediction(pred.acp[,2], Train$Revenue)
perf2 <- performance(pred2,"tpr","fpr")

plot(perf2, main="Curva ROC (in-sample) Árbol", colorize=T)

auc_test <- performance(pred2, measure = "auc")
auc_test@y.values[[1]] # 0.8805 pero no me importa 

# Curva ROC en TEST
pred.acp.Test = predict(arbol.podado,type='prob', newdata = Test)

pred2 <- prediction(pred.acp.Test[,2], Test$Revenue)
perf2 <- performance(pred2,"tpr","fpr")

plot(perf2, main="Curva ROC (in-sample) Árbol", colorize=T)

auc_test <- performance(pred2, measure = "auc")
auc_test@y.values[[1]] # 0.8474

###################### EXTENSIONES: ############################################

# 1. Optimizar el threshold: 

# Para que devuelva la probabilidad:
pred.acp.Prob= predict(arbol.podado,type='prob', newdata = Validation)

# Confusion matrix
table(Validation$Revenue, pred.acp.Prob[,2] > .5)

#Vemos que el default de class es 0.5, pero podriamos optimizarlo 


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
F2.est 
plot(thresh, F2.est, type = 'b', bty = 'n')
thresh.opt = thresh[max(which(F2.est==max(F2.est)))]
abline(v = thresh.opt[1] , col = 'red', lty = 2 )


#El parametro por default (0.5) maximiza F1, pero podría no haber sido el caso

pred.acp.Prob.test= predict(arbol.podado,type='prob', newdata = Test)

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
tasa_error_test # 13.75% USANDO THRESH.OPT. 
F2.est #71.18
