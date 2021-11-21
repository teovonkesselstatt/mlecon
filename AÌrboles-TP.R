# CART: Árboles de Clasificación
rm(list=ls())
#### Paquetes y librerias a instalar y cargar:########
require(rpart) || install.packages('rpart')
require(ROCR)  || install.packages('ROCR') 
require(rpart.plot) || install.packages('rpart.plot')

churn <- read.csv('Telco_churn.csv')
churn$X <- NULL
churn$state <- NULL

##########################################################
############     TRAIN, VALIDATION Y  TEST     ###########
##########################################################

id = sample(nrow(datos),0.8*nrow(datos))

Train=datos[id,] # Conjunto de training
Test=datos[-id,]  # Conjunto de test

dim(Train)
head(Train)
dim(Test)

# Vamos a hacer Validatoin Set Cross Validation, entonces separamos TRAIN en 2 grupos: 
id_validation = sample(nrow(Train),0.7*nrow(Train))

Train=Train[id_validation,] 
Validation = Train[-id_validation,]

######################
###     Árboles    ###
######################

# Repasamos los parametros del modelo rpart
help(rpart) #method: class y anova para regresión.
            #weights: igual que antes, podemos repesar a las observaciones

help(rpart.control) 
# minsplit: Numero minimo de observaciones que tiene que tener el nodo final
#           para hacer una nueva partición

# cp: "Complexity Parameter" 
#     Si no mejora el fit en "cp", no hace una nueva partición
#     EJ: en anova, si el R^2 no mejora mas de cp, no hace una nueva particion

# maxdepth: cantidad de nodos 
# xval: folds para CV

arbol.datos = rpart(churn ~ ., data=Train, 
                    method="class",    # 'anova' para regresión.
                    parms = list( split = "gini"),       # Métrica con la que determinan los cortes.
                    control= rpart.control(minsplit = 100,       # Cantidad minima de observaciones en nodo (antes de partir)
                                           xval = 8 ,            # cantidad de folds de validación
                                           cp = 0.0001,           # Umbral de mejora mínima (equivale a "alpha" en escala [0,1]).
                                           maxdepth = 10 )      # Longitud maxima del arbol.
                    )

rpart.plot(arbol.datos, type = 2, cex=0.75)

#REESCRIBÍ TODO HASTA ACA PERO NO CORRE PORQUE#
#"Error: unexpected symbol in:"         )rpart.plot""#


# En cada nodo terminal: 
# Predicción
# hat(P(churn=yes))
# % de datos 

#Interpretar: X nueva
# total_day_minutes = 150
# number_customer_service_calls = 3
# total_int_minutes = 15
# total_eve_minutes = 300
# voice_mail_plan_yes

#¿Qué predice el modelo para esta nueva X?



# Importancia de cada variable en la decisión:
arbol.datos$variable.importance # Importancia de cada covariable (feature).

importancia <- arbol.datos$variable.importance
nombres <- names(importancia)

df <- cbind(importancia, nombres)
zeros <- cbind(rep(0,16), nombres)
df<- data.frame(rbind(df, zeros))

df$importancia <- as.numeric(df$importancia)
df$nombres <- as.factor(df$nombres)

ggplot(df, aes(x=importancia, y=nombres))+geom_line(aes(colour=nombres, size=2), show.legend = F)



# Estimación por VC del tamaño optimo del árbol:
### Como "podar el árbol":
errores <-printcp(arbol.Churn) # Elegimos el valor de cp que minimiza el error 
which.min(errores[,4])
errores[which.min(errores[,4]),1]
# estimado por VC (columna xerror).


arbol.Churn.podado = prune(arbol.Churn,cp = errores[which.min(errores[,4]),1] )

rpart.plot(arbol.Churn.podado, type =2, cex=0.75) 


### Predicciones con el arbol:

pred= predict(arbol.datos.podado, type='class')
head(pred,3)


# Matriz de Confusion (datos de train):
table(pred,Train$churn)

1-sum(diag(table(pred,Train$churn)))/sum(table(pred,Train$churn))
#T.Error = 0.0625

# Curva ROC
library(ROCR)

# Curva ROC
pred.acp = predict(arbol.Churn.podado,type='prob', newdata = Train)

pred2 <- prediction(pred.acp[,2], Train$churn)
perf2 <- performance(pred2,"tpr","fpr")

plot(perf2, main="Curva ROC (in-sample) Árbol", colorize=T)


# Predicciones en TEST
pred.acp.Test = predict(arbol.Churn.podado,type='class', newdata = Test)
head(pred.acp.Test)

# Matriz de confusion sobre TEST
table(pred.acp.Test,Test$churn)
table(pred.acp.Test,Test$churn)/dim(Test)[1]*100

# Tasa de Error sobre TEST
tasa_error <- 1-sum(diag(table(pred.acp.Test,Test$churn)))/dim(Test)[1]

# Tasa de Error en Test 5.8% 
# (Similar a la muestra Train => el modelo esta bien calibrado!)

# Curva ROC en TEST
pred.acp.Test = predict(arbol.Churn.podado,type='prob', newdata = Test)

pred2 <- prediction(pred.acp.Test[,2], Test$churn)
perf2 <- performance(pred2,"tpr","fpr")

plot(perf2, main="Curva ROC (in-sample) Árbol", colorize=T)



### EXTENSIONES: 

# 1. Optimizar el threshold: 

# Cuando predecimos podemos pedirle 'class'
pred.acp.Val = predict(arbol.Churn.podado,type='class', newdata = Validation)
table(Validation$churn, pred.acp.Val)

# O podemos pedirle 'prob' para que devuelva la probabilidad:
pred.acp.Prob= predict(arbol.Churn.podado,type='prob', newdata = Validation)
# Confusion matrix
table(Validation$churn, pred.acp.Prob[,2] > .5)

#Vemos que el default de class es 0.5, pero podriamos optimizarlo 


#Optimizamos el Threshold: 
thresh = seq(0.01,0.7,length.out = 100)
thresh

# Optimizamos el threshold eligiendo el que maximiza F1
F1.est = c();
for(i in 1:length(thresh)){
  #Clasificamos a las observaciones
  y_hat = as.numeric(pred.acp.Prob[,2] >= thresh[i])
  TP = sum( y_hat== 1  & Validation$churn=='yes' )    # True  +/1/'yes'
  FP = sum( y_hat == 1 & Validation$churn=='no' )     # False +/1/'yes'
  FN = sum( y_hat == 0 & Validation$churn=='yes' )    # False -/0/'no'
  TN = sum( y_hat == 0 & Validation$churn=='no' )     # True  -/0/'no'
  prec = TP/(TP + FP) ; rec = TP/(TP + FN)
  F1.est[i] <- (1+1^2)*(prec*rec)/(1^2*prec + rec)*100
  print(i)
}
F1.est 
plot(thresh, F1.est, type = 'b', bty = 'n')
thresh.opt = thresh[which(F1.est==max(F1.est))]
abline(v = thresh.opt[1] , col = 'red', lty = 2 )


#El parametro por default (0.5) maximiza F1, pero podría no haber sido el caso



# 2. Optimizar el w (igual que antes, por class imbalance)

gamma = c(0,0.01,0.1,0.2,0.3,0.5, 0.6, 0.7,0.75, 0.8,0.85, 0.9) 
F2.gamma = c() # <- acá guardo el F4 de cada modelo estimado para cada valor de gamma (obviamente que se puede usar otra métrica)
for(i in 1:length(gamma)){
  #Definimos los pesos 
  w = ifelse(Train$churn=='yes',1+gamma[i],1-gamma[i]) 
  set.seed(1)
  arbol.Churn = rpart(churn ~ ., data=Train, 
                      method="class",    # 'anova' para regresión.
                      parms = list( split = "gini"),       # Métrica con la que determinan los cortes.
                      control= rpart.control(minsplit = 100,       # Cantidad minima de observaciones en nodo (antes de partir)
                                             xval = 8 ,            # cantidad de folds de validación
                                             cp = 0.0001,           # Umbral de mejora mínima (equivale a "alpha" en escala [0,1]).
                                             maxdepth = 10), # Longitud maxima del arbol.
                      weights = w
  )
  
  errores <-printcp(arbol.Churn) # Elegimos el valor de cp que minimiza el error 
  which.min(errores[,4])
  errores[which.min(errores[,4]),1]
  # "Podamos" el arbol
  arbol.Churn.podado = prune(arbol.Churn,cp = errores[which.min(errores[,4]),1] ) 
  
  #Una vez elegido el mejor arbol para w, calculamos la métrica (F2 en este caso)
  #Para eso, predecimos sobre Validation:
  
  y_hat = predict(arbol.Churn.podado,type='class', newdata = Validation)
  #Calculamos el error:
  TP = sum( y_hat== 'yes'  & Validation$churn=='yes' )    # True  +/1/'yes'
  FP = sum( y_hat == 'yes' & Validation$churn=='no' )     # False +/1/'yes'
  FN = sum( y_hat == 'no' & Validation$churn=='yes' )    # False -/0/'no'
  TN = sum( y_hat == 'no' & Validation$churn=='no' )     # True  -/0/'no'
  prec = TP/(TP + FP) ; rec = TP/(TP + FN)
  F2.gamma[i] <- (1+2^2)*(prec*rec)/(2^2*prec + rec)*100
  print(i)
}

print(F2.gamma)

plot(gamma, F2.gamma, type = 'b')
abline(v = gamma[which.max(F2.gamma)] , col = 'red', lty = 2 )
#Gamma opt=0.5

# Veamos la performance en TEST:
## Primero tenemos que entrenar el arbol con los parametros optimizados:
set.seed(123) 
gamma= gamma[which.max(F2.gamma)]

w = ifelse(Train$churn=='yes',1+gamma,1-gamma) 
set.seed(1)
arbol.Churn = rpart(churn ~ ., data=Train, 
                    method="class",    # 'anova' para regresión.
                    parms = list( split = "gini"),       # Métrica con la que determinan los cortes.
                    control= rpart.control(minsplit = 100,       # Cantidad minima de observaciones en nodo (antes de partir)
                                           xval = 8 ,            # cantidad de folds de validación
                                           cp = 0.0001,           # Umbral de mejora mínima (equivale a "alpha" en escala [0,1]).
                                           maxdepth = 10), # Longitud maxima del arbol.
                    weights = w
)

# Elegimos el mejor CP: 
errores <-printcp(arbol.Churn) # Elegimos el valor de cp que minimiza el error 
which.min(errores[,4])
errores[which.min(errores[,4]),1]
# "Podamos" el arbol
arbol.Churn.podado = prune(arbol.Churn,cp = errores[which.min(errores[,4]),1] ) 

#El arbol optimo: 
rpart.plot(arbol.Churn.podado, type =2, cex=0.75) 


#METRICAS EN TEST:
pred.acp.Test = predict(arbol.Churn.podado,type='class', newdata = Test)
head(pred.acp.Test)

# Matriz de confusion sobre TEST
table(pred.acp.Test,Test$churn)
table(pred.acp.Test,Test$churn)/dim(Test)[1]*100

# Tasa de Error sobre TEST
tasa_error <- 1-sum(diag(table(pred.acp.Test,Test$churn)))/dim(Test)[1]

# Tasa de Error en Test 8,8% sube
# Pero elegimos el parametro para optimizar el F2, no la tasa de error

