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

sum(is.na(datos))

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
mean(datos[,3], na.rm = T)

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
### Cuando se hace eso tener en cuenta en que posicion queda la variable Y (Spoiler:207)

