# install.packages("FNN")
library(FNN)
####################################################################
#### Modelo de regresión NO lineal para datos de demanda de bikes: #
####################################################################
rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica
### Leemos datos externos con el comando "read.table" (ojo con el working directory!):
datos = read.table('bikes.csv',
                   sep=',',header=T,dec='.',na.strings = "NA")

str(datos)    # Visualizar las propiedades de las variables leídas.

head(datos,2) # Visualizar las primeras filas del data set.

#### Diccionario de variables:
# instant: record index
# dteday : date
# season : season (1:winter, 2:spring, 3:summer, 4:fall)
# yr : year (0: 2011, 1:2012)
# mnth : month ( 1 to 12)
# hr : hour (0 to 23)
# holiday : weather day is holiday (1) or not (0).
# weekday : day of the week
# workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# weathersit: 
#1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# hum: Normalized humidity. The values are divided to 100 (max)
# windspeed: Normalized wind speed. The values are divided to 67 (max)
# casual: count of casual users
# registered: count of registered users
# cnt: count of total rental bikes including both casual and registered

attach(datos) # adjuntamos las variables del data frame en R.
# facilita que puedas "llamar" a las variables por su nombre.
# de lo contrario, habria que utilizar datos$variable

#####################################################################
# Vamos a trabajar solo con 1 variable para poder visualizar        #
# Ejercicio: Regresión lineal (benchmark) vs K-nn para predecir     #
#            la demanda de bicis (cnt)en función de la temperatura. #
#####################################################################


########################################################
########## Validation set approach #####################
########################################################
# 1 Separar los datos en TRAIN y TEST:
n = length(cnt)
set.seed(12345)
sample.id = sample(n,0.75*n) # Asigno aleatoriamente 75% de los datos a train y 25% a test
# 548 instancias para TRAIN y 183 instancias para TEST. 

train.X = matrix(datos[sample.id,c(10)],ncol=1)   
train.Y = matrix(datos[sample.id,c(16)],ncol=1)   

test.X = matrix(datos[-sample.id,c(10)],ncol=1)   
test.Y = matrix(datos[-sample.id,c(16)],ncol=1)   

#@@@ Benchmark:
reglin = lm(cnt~temp, subset = sample.id)
summary(reglin)

pred.reglin = predict(reglin, newdata = as.data.frame(temp))[-sample.id]
log(sum((test.Y - pred.reglin)^2)/ 183) # log del ECM (14.73) 

#@@@ "Estimación" de "k" (parámetro de complejidad): 
valores.k = c(5,10,25,50,100,200,400,500)
ECM = c() # Aquí guardaremos los ECM
for(i in 1:length(valores.k)){
  kNN.Univ = knn.reg(train = train.X , test = test.X, y = train.Y , 
                     k = valores.k[i]) # <- mira esta línea!
  ECM[i] = sum((test.Y - kNN.Univ$pred)^2)/183 
}


plot(valores.k,log(ECM), type='b',pch=20, main ='')

valores.k[which(ECM==min(ECM))]

log(ECM[which(ECM==min(ECM))]) # 14.6

############################################################
########## k--fold cross validation:   #####################
############################################################
# Permutación aleatoria de los datos
set.seed(123)
datos2 <- cbind(temp,cnt)
datos2 <- datos2[sample(nrow(datos)),] 

# Creo 10 'folds'
folds <- cut(seq(1,nrow(datos2)),breaks=10,labels=FALSE)
folds # Un indicador de cada uno de los folds

ECM = c(); # Para cada valor del parámetro computamos el promedio de 10 estimaciones del ECME
ECM.sd = c();   # Para cada valor del parámetro computamos el sd del estimador del ECME
for(i in 1:8){ # i recorre los valores del parámetro k.
  ECM.kfold = c()
  for(j in 1:10){ # j-recorre los folds
    index.train = which(folds!=j) # j recorre los folds.
    index.test  = which(folds==j) # j recorre los folds.
    
    kNN.Univ = knn.reg(train = as.matrix(datos2[index.train,1]) , 
                       test = as.matrix(datos2[index.test ,1]) , 
                       y = as.matrix(datos2[index.train,2]) , 
                       k = valores.k[i]) # i recorre valores del parámetro k
    ECM.kfold[j] = sum((datos2[index.test,2] - kNN.Univ$pred)^2)/nrow(datos2[index.test,])
  }
  
  ECM[i]   = mean(ECM.kfold); 
  ECM.sd[i]= sd(ECM.kfold);
  print(i)  
}

ECM
ECM.sd

# Plot de mi primer selección de parámetros por VC (que emoción!) 
plot(valores.k,log(ECM),type='b',pch=20, main='10-fold VC') # :)

valores.k[which(ECM==min(ECM))]

log(ECM[which(ECM==min(ECM))]) # 14.53

################################################################################## FIN.

# Modelo lineal 
reglin = lm(cnt~temp)
summary(reglin)

#### Comparativa gráfica de la performance de los 2 modelos:
plot(temp,cnt,pch=20) 
abline(reglin, col='red', lwd = 3)

# kNN con k = 100
kNN.Univ = knn.reg(train = temp , test = NULL, y = cnt, k = 100)
predicciones.ordenadas = cbind(temp, kNN.Univ$pred) 
predicciones.ordenadas = predicciones.ordenadas[order(predicciones.ordenadas[,1], 
                                                      decreasing = TRUE),] 
points(predicciones.ordenadas, type='l', col = 'blue', lwd = 3)

log(sum(reglin$residuals^2)/n)   # reg lineal.
log(sum((kNN.Univ$pred-cnt)^2)/n) # kNN con VC.

sum(reglin$residuals^2)/sum((kNN.Univ$pred-cnt)^2) # k-nn es 11% más preciso q rl.
##################################################################################






###############################################
# EJERCICIO: Extender a todas las covariables #
###############################################

rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica
### Leemos datos externos con el comando "read.table" (ojo con el working directory!):
datos = read.table('bikes.csv',
                   sep=',',header=T,dec='.',na.strings = "NA")
attach(datos)

# Chequeamos que no haya missings:
sum(is.na(datos)==T)

# Removemos variables:
datos$instant <- NULL # Es una ID temporal, la podemos dejar si consideramos que la serie tiene tendencia.
datos$dteday <- NULL # No esta bien codeada, en todo caso no aporta mas info que instant y el resto de variables temporales.
datos$registered <- NULL # cnt = registered + casual
datos$casual <- NULL
datos$atemp <- NULL # Colinealidad con temp. Para verlo, usar cor(temp, atemp) antes de borrarla.

# Asignamos variables categóricas como factores
summary(lm(cnt~., data=datos))

datos$season <- as.factor(datos$season)
datos$yr <- as.factor(datos$yr)
datos$mnth <- as.factor(datos$mnth)
datos$holiday <- as.factor(datos$holiday)
datos$weekday <- as.factor(datos$weekday)
datos$workingday <- as.factor(datos$workingday)
datos$weathersit <- as.factor(datos$weathersit)

summary(lm(cnt~., data=datos))
#La variable workingday tiene problemas de colinealidad, podemos eliminarla
datos$workingday <- NULL
summary(lm(cnt~., data=datos)) # Se resolvió el problema


#######################################################
########    Benchmark: Regresión lineal   #############
#######################################################

# 1 Separar los datos en TRAIN y TEST:
n = length(cnt)
set.seed(123)
sample.id = sample(n,0.75*n) # Asigno aleatoriamente 75% de los datos a train y 25% a test
# 548 instancias para TRAIN y 183 instancias para TEST. 

train.X = matrix(datos[sample.id,-c(10)])   
train.Y = matrix(datos[sample.id,c(10)])   

test.X = matrix(datos[-sample.id,-c(10)],ncol=1)   
test.Y = matrix(datos[-sample.id,c(10)],ncol=1)   

#@@@ Benchmark:
reglin = lm(cnt~ ., subset = sample.id, data = datos)
summary(reglin)

pred.reglin = predict(reglin, newdata = as.data.frame(datos))[-sample.id]
ECM_reglin = log(sum((test.Y - pred.reglin)^2)/ 183) # log del ECM (13.43196) 

plot(pred.reglin, datos$cnt[-sample.id], pch=20, xlab="Estimación", ylab="Datos observados",
     main="Fit de la Regresión Lineal")
abline(a=0, b=1, col='red')

############################################################################
##########  PREPARACIÓN DE LOS DATOS Y VALIDACION CRUZADA  #################
############################################################################

# Permutación aleatoria de los datos
set.seed(12345)
datos2 <- datos
datos2 <- datos2[sample(nrow(datos)),] 

# Para trabajar con variables categóricas hay que aplicar transformaciones
# En este caso, hacemos "one hot encoding", es decir, creamos dummies para todas.
# Para hacerlo facilmente utilizamos la libreria "fastDummies"
# install.packages("fastDummies")
library(fastDummies)

# dummy_cols: Crea automáticamente dummies para cada variable del dataframe codeada como Factor o Character
# El parametro "remove_first_dummy" significa que se crean k-1 dummies para k categorías de la variable.
# La función no elimina las variables originales, con lo cual hay que borrarlas.
datos2 <- fastDummies::dummy_cols(datos, remove_first_dummy = T) 
datos2$season <- NULL
datos2$yr <- NULL
datos2$mnth <- NULL
datos2$holiday <- NULL
datos2$weekday <- NULL
datos2$weathersit <- NULL
datos2$workingday <- NULL


#Escalamos los datos: 
datos2_scaled <- as.data.frame(scale(datos2[,c(1:3)]))
datos2_scaled <- cbind.data.frame(datos2_scaled, datos2[, 4:28])



# Creo 10 'folds'
folds <- cut(seq(1,nrow(datos2_scaled)),breaks=10,labels=FALSE)
folds # Un indicador de cada uno de los folds

valores.k = c(5,10,20,30, 40,45, 50,55, 60, 70, 80, 90, 100,300, 400, 500) 
#Puedo ajustar la grilla de valores de k para que tenga valores cercanos a 50 y volver a elegir con "zoom"


ECM = c(); # Para cada valor del parámetro computamos el promedio de 10 estimaciones del ECME
ECM.sd = c();   # Para cada valor del parámetro computamos el sd del estimador del ECME
for(i in 1:length(valores.k)){ # i recorre los valores del parámetro k.
  ECM.kfold = c()
  for(j in 1:10){ # j-recorre los folds
    index.train = which(folds!=j) # creamos los datos de train del fold j
    index.test  = which(folds==j) # creamos los datos de test del fold j
    
    kNN.Univ = knn.reg(train = as.matrix(datos2_scaled[index.train,-c(4)]) , 
                       test = as.matrix(datos2_scaled[index.test, -c(4)]) , 
                       y = as.matrix(datos2_scaled[index.train,c(4)]) , 
                       k = valores.k[i]) # i recorre valores del parámetro k
    ECM.kfold[j] = sum((datos2_scaled[index.test,c(4)] - kNN.Univ$pred)^2)/nrow(datos2_scaled[index.test,])
  }
  
  ECM[i]   = mean(ECM.kfold); 
  ECM.sd[i]= sd(ECM.kfold);
  print(i)  
}

ECM
ECM.sd

# Plot de mi primer selección de parámetros por VC (que emoción!) 
plot(valores.k,log(ECM),type='b',pch=20, main='10-fold VC') # :)

valores.k[which(ECM==min(ECM))] # k = 55

ECM_knn = log(ECM[which(ECM==min(ECM))]) 
# 14.56904, el agregado de variables mejora el ECM promedio en la 10-fold CV en
# relación con el modelo con un solo regresor. 

#Sin embargo, si lo comparamos con la regresión lineal: 
ECM_knn/ECM_reglin #knn es peor que la regresión lineal




## Una vez que elegimos el parámetro (40) podemos correr la regresión
knn_reg = knn.reg(train = as.matrix(datos2_scaled[index.train,-c(4)]) , 
                   test = as.matrix(datos2_scaled[index.test, -c(4)]) , 
                   y = as.matrix(datos2_scaled[index.train,c(4)]) , 
                   k = 55) 

pred_knn <- knn_reg$pred



#Podemos hacer un gráfico de las predicciones vs test. Queremos que se parezca lo más posible a la recta de 45

plot(pred.reglin, datos$cnt[-sample.id], pch=20, xlab="Estimaciones", ylab="Datos observados",
     main="Fit de la Regresión Lineal vs K-vecinos")
abline(a=0, b=1, col='red')
points(pred_knn, datos2_scaled[index.test, c(4)], pch=20, col='blue')
legend(1000, 8000,legend=c('Regresión lineal', 'K-vecinos'), col=c('black', 'blue'),pch=20, cex=0.8)





######################################################
################ Clasificación #######################
######################################################

#Si nos encontramos con un problema de clasificación, el algoritmo es el mismo. 
#La única diferencia es utilizamos la función knn en vez de knn.reg
# Además, hay que agregar prob = TRUE

#Para hacer Cross Validation tenemos que tener una métrica para el error. 
#Lo vamos a ver más adelante 














