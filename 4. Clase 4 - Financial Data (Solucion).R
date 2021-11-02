# Origen de los datos: https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018 
#                      https://financialmodelingprep.com/
rm(list=ls())
setwd("C:/Users/teovo/Google Drive/DI TELLA/8vo semestre/Machine Learning/data")
datos = read.csv('2018_Financial_Data.csv', header = T, sep = ",", na.strings = "")

# 225 variables
# 4392 obs

# Variables de respuesta: PRICE para regresion (Class para clasificacion, clase que viene)
head(datos[,224:225], 3)

#OBJETIVO: predecir los precios de acciones en base a indicadores financieros

# Problemas que tiene el dataset


#################
###### NAs ######
#################

# Alternativas:
# 1) Si tengo pocos NA puedo imputar: media, mediana, kNN 
# 2) Si tengo muchos NA en una variable, puedo descartar la variable.
# 3) Si tengo muchos NA en una observacion, puedo descartarla. 

# Eliminamos columnas con > 1000 NAs
colnas = c()
for (i in 1:223){
  colnas[i]=length(which(is.na(datos[,i])))}

colnas # Vemos cuantos NA hay en cada columna.

datos[, which(colnas>1000)] <- NULL # Eliminamos columnas con mas de 1000 NAs

# Se eliminaron 26 variables que tenían casi 1/4 de los datos NA


# Reemplazamos resto de NAs por medias de resto de observaciones de cada columna.
#Una alternariva: en vez de reemplazar por la media usar librerías que estimen los missings 
#Algunas son: 
# AMELIA: https://cran.r-project.org/web/packages/Amelia/Amelia.pdf
# Impute.knn: https://www.rdocumentation.org/packages/impute/versions/1.46.0/topics/impute.knn

for (i in 2:197){
  datos[which(is.na(datos[,i])),i] = mean(datos[,i], na.rm = T) #Reemplazo NAs por medias
  print(i)
}

mean(datos[,2], na.rm = T)
mean(datos[,2], na.rm = T)
# Repetimos conteo de NA para chequear.
colnas = c()
for (i in 1:223){
  colnas[i]=length(which(is.na(datos[,i])))}

colnas # 0 para todas las variables.

# Outliers
# Por ejemplo:
#Variable de resultado:

# Para ver outliers algo muy útil es ver los boxplots: 
## Las observaciones que están fuera de los "bigotes" son outliers 

# Como debería ser:
normal <- rnorm(4000,mean= 0,sd= 1)
boxplot(normal)

# Como son: 
boxplot(datos[,2])
boxplot(datos[,3])
boxplot(datos[,4])
boxplot(datos[,5])

boxplot(datos[,198]) #la variable de respuesta


# Censuramos valores por debajo del cuantil 1% y por encima del cuantil 99% 
# En cada loop calculamos los cuantiles de la columna y reemplazamos outliers.
for (i in 2:196){
  q0.01 = quantile(datos[,i], 0.01)  
  q0.99 = quantile(datos[,i], 0.99) 
  datos[which(datos[,i]<q0.01),i] = q0.01
  datos[which(datos[,i]>q0.99),i] = q0.99
}

# Saltamos la variable 197 porque es "character". Lo hacemos para la variable de respuesta: 

q0.01 = quantile(datos[,198], 0.01)  
q0.99 = quantile(datos[,198], 0.99) 
datos[which(datos[,198]<q0.01),198] = q0.01
datos[which(datos[,198]>q0.99),198] = q0.99

# Ahora:
boxplot(datos[,2])
boxplot(datos[,3])
boxplot(datos[,4])
boxplot(datos[,5]) #Capaz que todavia hay muchos outliers, podemos cambiar los cuantiles
boxplot(datos[,198])

#########
# Tarea #
#########

# Utilice algún método de regularización de modelos lineales (Ridge, Lasso, Elastic Nets) para predecir 
# la variación de la acción en el año siguiente (Columna 198 del dataset "depurado").
# Elija el nivel de regularización (lambda) utilizando algún método de validación cruzada visto en clase.

# Ayuda: Recuerde que las columnas 2 y 197 son variables categóricas. 

#Se puede omitir la 2 (datos$X <- NULL) 

#La 197 hace referencia al sector y puede ser informativo, podemos hacer modelmatrix o 
# one hot encoding como vimos en clases anteriores. 




### Solucion: 
library(glmnet)

#Elimino la primera variable: (ahora la Y es la columna 197)
datos$X <- NULL
head(datos[,195:198]) #La Y es X2019.PRICE.VAR....

#Hacemos la model matrix
modmat <- model.matrix( ~ ., datos)
ncol(modmat) #Tenemos 208 variables (se expandieron la categorias)
modmat[,207]==datos$X2019.PRICE.VAR....
#Ahora la variable de interes es la 207

#Separamos en train y test
set.seed(1) ; id = sample(nrow(datos),0.8*nrow(datos))
x_train = modmat[id, -c(207)] 
x_test = modmat[-id,-c(207)]
y_train = modmat[id, c(207)] 
y_test = modmat[-id, c(207) ]

##############################
#### RIDGE (alpha = 0)  ######
##############################

grid.l =exp(seq(10 , -10 , length = 100)) # El paquete nos permite estimar todo el vector de lambdas

plot(grid.l, type = 'b') 

# Estamos entrenando 100 modelos (1 para cada valor de lambda)
ridge.reg = glmnet(x = x_train , 
                   y = y_train,
                   family = 'gaussian',  # modelo de regresión lineal.
                   alpha = 0,           # Ridge.
                   lambda = grid.l,      # Grilla para lambda.
                   standardize=TRUE) 

# ¿Cómo cambia cada coeficiente con respecto a lambda?
dev.off()
plot(ridge.reg, xvar = "lambda", label = TRUE, xlab = expression(log(lambda)))

##########################################################################
######### Aprendiendo el mejor valor de lambda por VC ("5 FOLDS")   ######
##########################################################################
set.seed(1)
cv.out = cv.glmnet(x_train, y_train, lambda = grid.l, alpha = 0, nfolds = 5)
cv.out

# Grafica de ECM como funcion de lambda. En ese caso, nos pone intervalos de ECM para cada lambda y media.
dev.off()           # Verás una gráfica similar solo que tendras para cada valor de lambda
plot(cv.out)        # una estimación de la distribución del ecm (basada en los 5 folds del método de VC).

# Valores "criticos" de lambda
cv.out$lambda.min   # Primera linea desde la izq: lambda que minimiza log(ecm)
cv.out$lambda.1se   # Segunda linea desde la izq: lambda mas grande tal que log(ecm) se mantiene en 1se

bestlam = cv.out$lambda.min

# Una vez que estimamos el mejor valor de lambda (bestlam) calculamos ECM out:
ridge.pred = predict(ridge.reg, s = bestlam , newx = x_test)
ecm.ridge = sum((ridge.pred - y_test)^2)/ nrow(x_test) 
log(ecm.ridge)#7.085886



##############################
#### LASSO (alpha = 1)   #####
##############################

grid.l =exp(seq(10 , -10 , length =100))
LASSO.reg = glmnet(x = x_train , 
                   y = y_train,
                   family = 'gaussian', # Regresión. 
                   alpha = 1,           # LASSO 
                   lambda = grid.l,     # Grilla para lambda.
                   standardize=TRUE)    # Flag TRUE por defecto.

# Notar que este modelo "selecciona variables".
dev.off()
plot(LASSO.reg, xvar = "lambda", label = TRUE, xlab = expression(log(lambda)))


#Validacion cruzada:
cv.out = cv.glmnet(x_train, y_train, 
                   alpha = 1, 
                   nfolds = 5, 
                   type.measure = c("mse"), # con métrica cross-validamos.
                   lambda = grid.l)  

plot (cv.out)
bestlam = cv.out$lambda.min
bestlam

# Una vez que estimamos el mejor valor de lambda (bestlam) calculamos ECM out:
LASSO.pred = predict(LASSO.reg, s = bestlam , newx = x_test)
ecm.lasso = sum((LASSO.pred - y_test )^2)/nrow(x_test)
log(ecm.lasso) #7.075484



#################################
#### ENet (alpha in (0,1))  #####
#################################

# Promedio entre la distancia de ridge y de lasso
# Antes elegíamos solo lambda
# Ahora también hay que elegir alpha (cómo ponderamos el promedio)


grid.l =exp(seq(10 , -10 , length =100))
ENet.reg = glmnet(x = x_train , 
                  y = y_train,
                  family = 'gaussian', # Regresión. 
                  alpha = 0.5,         # Red elástica con alpha = 1/2 
                  lambda = grid.l,     # Grilla para lambda.
                  standardize=TRUE)    # Flag TRUE por defecto.


# Valicacion Cruzada:
ecm.enet = c()
alpha.grid = c(0.1,0.25,0.5,0.75,0.9) # Grilla para el parámetro alpha.
bestlam_EN = c()
for(i in 1:5){
  set.seed(123)
  cv.out = cv.glmnet(x_train, y_train, alpha = alpha.grid[i], nfolds = 10) # para alpha fijo cross valido lambda.
  ENet.reg = glmnet(x = x_train , 
                    y = y_train,
                    family = 'gaussian',             
                    alpha = alpha.grid[i],          # Valor de alpha. 
                    lambda = cv.out$lambda.min)     # Valor optimo de lambda estimado por VC.
  
  bestlam_EN[i] = cv.out$lambda.min
  ENet.pred = predict(ENet.reg, newx = x_test)
  ecm.enet[i] = sum((ENet.pred - y_test )^2) / nrow(x_test) # computo el valor del ecm para (alpha, lambda^*).
  print(i)
}

plot(alpha.grid, ecm.enet, type = 'b') # De acá selecciono alpha^* Ojo, resultados sensibles a folds!

which.min(ecm.enet) #Se minimiza en 3
alpha.grid[3]

#Tambien se hizo CV de lambda: 
bestlam_EN[3]


ENet.reg = glmnet(x = x_train, 
                  y = y_train,
                  family = 'gaussian',             
                  alpha = alpha.grid[3],          # Valor de alpha. 
                  lambda = bestlam_EN[3])     # Valor optimo de lambda estimado por VC.

ENet.pred = predict(ENet.reg, newx = x_test)
ecm.enet = sum((ENet.pred - y_test )^2) / nrow(x_test)
log(ecm.enet) #  7.074949


#Comparo los modelos: 
errores = cbind(ecm.enet, ecm.lasso, ecm.ridge)
which.min(errores) # Elastic Nets minimiza



