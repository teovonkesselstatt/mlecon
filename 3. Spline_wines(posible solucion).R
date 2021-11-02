# ------------- Splines multivariantes y selección de modelos -------------#
# Objetivos: 
#           -1er: Predecir la 'calidad' del vino. 
#           -2do: Identificar características (covariables) relevantes que determinar la calidad.
# Datos: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
rm(list=ls()); dev.off()
setwd("C:/Users/teovo/Google Drive/DI TELLA/8vo semestre/Machine Learning/data")
data = read.table('winequality-white.csv',sep=';',
                   header=T,dec='.',na.strings = "")

require(tidyverse) || install.packages('tidyverse')
library(tidyverse)


dim(data) # 11 covariables y una variable a predecir 'quality'. 

str(data) # Las información cualitativa puede introducirse utlizando dummy's.

head(data,2)
sum(is.na(data)) # NO hay datos faltantes.

require(psych) || install.packages('psych'); library(psych)
x11()
pairs.panels(data, cex = 1, method = "pearson", hist.col = "#00AFBB",
             smooth = TRUE, density = TRUE, ellipses = FALSE )

# Se ven algunas observaciones que aprentan ser atípicas y podrían influír en el
# modelo. Eventualmente las podríamos quitar del data set o darle menos peso en
# el proceso de aprendizaje de los parámetros del modelo. 


# Discutmaos como fitear y seleccionar modelos aditivos.

set.seed(1)
id.train <- sample(dim(data)[1], 3500) 
k = 3 # Parámetros general con el que controlo  la cantidad de nodos en "cada coordenada".

# Creamos una matriz de diseño "X" que contiene las bases splines en cada una de las covariables.
require(splines) || install.packages('splines'); library(splines)

X = matrix(ncol = 0, nrow = dim(data)[1])

?bs

for(i in 1:11){ X = cbind(X, bs(x = as.matrix(data[,i]), df = 3, intercept = F))}
dim(X) # Notas que p/n "se hace más grande"  (X tiene 33 columnas)
# head(X,2) # Matriz "X" del modelo: y = Xb + e

y = data[,12]

# Re-ordeno los datos en un solo data frame reg.data = [y,X]
reg.data = data.frame(y = y, X) 
bs.benchmark <- lm(y ~ ., subset = id.train, data = reg.data)
summary(bs.benchmark)

pred.val <- predict(bs.benchmark, newdata = reg.data[-id.train,-1])
plot(pred.val, y[-id.train], pch = 20, xlab = 'Predicciones', ylab = 'Observados', main = 'Validación')
sum( (pred.val - y[-id.train])^2 )/ nrow(reg.data[-id.train,-1]) #  ECM = 0.5060756

# Intervalos de confianza para las predicciones:
pred.ci <- predict(bs.benchmark, newdata = reg.data[-id.train,-1], 
                   interval="confidence", level = 0.95)
head(pred.ci,3) # Este es un output relevante a la hora de diseñar 
# 'acciones perscriptivas' con las predicciones del modelo.

################################################################################
# Selección de modelo (optimización del hiperparámetro 'k')          ###########
################################################################################
# Hacemos 'validation-set approach' (otras estrategias de VC son válidas)
k = c(3:15); ecm.val = c()

for(j in 1:length(k)){ #Recorremos los valores de k 
  
  X = matrix(ncol = 0, nrow = dim(data)[1]) #Generamos una matriz vacía que luego vamos a construir en matriz de diseño. 
  
  for(i in 1:11){ X = cbind(X, bs(x = as.matrix(data[,i]), #Para cada uno de los 11 features genera la matriz de diseño
                                  df = k[j], # Toma los valores del vector "k" (Estamos haciendo CV sobre los grados de libertad)
                                  intercept = F))}
  
  reg.data = data.frame(y = data[,12], X) 
  
  bs.reg <- lm(y ~ ., subset = id.train, data = reg.data)
  
  pred.val <- predict(bs.reg, newdata = reg.data[-id.train,-1])
  ecm.val[j] = sum( (pred.val - y[-id.train])^2 )/ 1398 
}
ecm.val

plot(k, ecm.val, ylab = 'ECM sobre validación', type = 'b')

which.min(ecm.val) #5 grados de libertad
ecm.val[5] ; k[5] # ECM más pequeño sobre el conjunto de validación.





#-------------------------------------------------------------------------END.

# Mientras más no lineal sea la relación entre las 'X' y la 'y', más diferencia
# notarás entre la performance del modelo con splines y la regresión lineal.