setwd("C:/Users/teovo/Google Drive/DI TELLA/8vo semestre/Machine Learning/mlecon")

### Cargamos las librerías:

require(glmnet)|| install.packages('glmnet')  #regularización de modelos lineales
require(ROCR)  || install.packages('ROCR')
require(ggplot2) || install.packages('ggplot2')
library(glmnet)
library(ROCR)
library(ggplot2)
require(FNN) || install.packages('FNN')
library(FNN)

rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica

datos = read.table('online_shoppers_intention.csv',
                   sep=',',header=T,dec='.',na.strings = "N/A")

######################################################
################## Analisis Exploratorio #############
######################################################

### MISSINGS:
sum(is.na(datos)==TRUE)  #No hay missings

# Hago que todos los chr sean factor
datos$Revenue <- gsub(FALSE, 0, datos$Revenue)
datos$Revenue <- gsub(TRUE, 1, datos$Revenue)
datos$Weekend <- gsub(TRUE, 1, datos$Weekend)
datos$Weekend <- gsub(FALSE, 0, datos$Weekend)

datos$Month <- factor(datos$Month, levels = c("Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), ordered = TRUE)
datos$OperatingSystems <- factor(datos$OperatingSystems)
datos$Browser <- factor(datos$Browser)
datos$Region <- factor(datos$Region)
datos$TrafficType <- factor(datos$TrafficType)
datos$VisitorType <- factor(datos$VisitorType)
datos$Revenue <- factor(datos$Revenue)
datos$Weekend <- factor(datos$Weekend)

# Regresión robusta: ###########################################################

library(robust)
attach(datos)
r.weights <- glmRob(Revenue ~ ExitRates + Administrative + Administrative_Duration + Informational + Informational_Duration + ProductRelated + ProductRelated_Duration, family = binomial(), data = datos, x = TRUE, y = TRUE)
datos$weights = r.weights$weights
which(r.weights$weights < 0.1) # outliers!
d = ncol(datos)
datos = datos[,c(1:(d-2), d,d -1)]

write.csv(datos,"datos.csv", row.names = FALSE)

# Vemos cuántas observaciones de cada categoría hay:

### VARIABLES CATEGÓRICAS
g1 <- ggplot(data=datos, aes(Month))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!
g1
table(datos$Month)

g2 <- ggplot(data=datos, aes(VisitorType))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!
g2
table(datos$VisitorType)

g3 <- ggplot(data=datos, aes(Weekend))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!
g3
table(datos$Weekend)

g4 <- ggplot(data=datos, aes(OperatingSystems))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!
g4
table(datos$OperatingSystems)

g5 <- ggplot(data=datos, aes(Browser))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!
g5
table(datos$Browser)

g6 <- ggplot(data=datos, aes(Region))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!
g6
table(datos$Region)

g7 <- ggplot(data=datos, aes(TrafficType))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!
g7
table(datos$TrafficType)

# Analisis de variables numericas
g8 <- ggplot(data=datos, aes(SpecialDay, color = Revenue, fill = Revenue))+
  geom_density(alpha = 0.1)
g8

g9 <- ggplot(data=datos, aes(ProductRelated_Duration, color = Revenue, fill = Revenue))+
  geom_density(alpha = 0.1)
g9

g10 <- ggplot(data=datos, aes(ProductRelated, color = Revenue, fill = Revenue))+
  geom_density(alpha = 0.1)
g10

# ORDENARLOS EN EL MISMO ESPACIO:
require(gridExtra) || install.packages('gridExtra')
library(gridExtra)

grid.arrange(g1, g2, g3, g4,  ncol=2)
grid.arrange(g5, g6, g7, g8, ncol=2)
grid.arrange(g9, g10, g11, g12, ncol=2)

## CORRELACION ENTRE VARIABLES (Graficos lindos de correlaciones)
library(corrplot)

datos.numeric = datos[ , purrr::map_lgl(datos, is.numeric)]
corr.matrix <- cor(datos.numeric)
x11(); corrplot(corr.matrix, main="\nCorrelation Plot for Numerical Variables", method="number")

###Sacando Outliers###
  
datos$log = log(datos$Administrative_Duration + 1)
boxplot(datos$log)

boxplot(datos$Revenue)
plot(datos$Weekend,datos$Revenue)

plot(datos$OperatingSystems,datos$Administrative_Duration)

datos$administrative1 <- datos[datos$Administrative ==0,]

sub.set <- subset(datos, !(datos$Administrative_Duration == 0)) # saco todas las observaciones donde Administrative duration es 0
sub.set$log = log(sub.set$Administrative_Duration) # les saco el log
boxplot(sub.set$log) # hago un boxplot con esos datos

PERO NO HAGO NADA AL RESPECTO??

#Reescalando los outliers superiores#
q0.99 = quantile(datos$Administrative_Duration, 0.99) 
datos$Administrative_Duration[which(datos$Administrative_Duration>q0.99)] = q0.99

#Volviendo a graficar para chequear que nos da bien la reescalación#
datos$log = log(datos$Administrative_Duration+1)
boxplot(datos$log)

boxplot(datos$Administrative)

boxplot(datos$Informational)
q0.99 = quantile(datos$Informational, 0.99) 
datos$Informational[which(datos$Informational>q0.99)] = q0.99
boxplot(datos$Informational)

boxplot(datos$ProductRelated)
q0.99 = quantile(datos$ProductRelated, 0.99) 
datos$ProductRelated[which(datos$ProductRelated>q0.99)] = q0.99
boxplot(datos$ProductRelated)

boxplot(datos$ProductRelated_Duration)
q0.99 = quantile(datos$ProductRelated_Duration, 0.99) 
datos$ProductRelated_Duration[which(datos$ProductRelated_Duration>q0.99)] = q0.99
boxplot(datos$ProductRelated_Duration)

#Ploteo Conjunto de Variables

plot(datos$ProductRelated_Duration,datos$ProductRelated, col=datos$Revenue)
plot(datos$SpecialDay,datos$ProductRelated_Duration, col=datos$Revenue)
plot(datos$Month,datos$SpecialDay, col=datos$Revenue)
plot(datos$Administrative_Duration,datos$Month,col=datos$Revenue)
plot(datos$OperatingSystems,datos$Administrative_Duration,col=datos$Revenue)
plot(datos$Browser,datos$Administrative_Duration,col=datos$Revenue)
plot(datos$Informational_Duration,datos$Administrative_Duration,col=datos$Revenue)
plot(datos$ProductRelated_Duration,datos$Administrative_Duration,col=datos$Revenue)

############## 3/12/21 ##############

#viendo un poco más distribución de variables.
#ni ahi hay normalidad.
hist(datos$Administrative)
hist(datos$Administrative_Duration)
hist(datos$ProductRelated)
hist(datos$ProductRelated_Duration)
hist(datos$SpecialDay)

#metiendo variables nuevas
#1
mean(datos$PageValues)
hist(datos$PageValues)
datos$PageValuesH = ifelse(datos$PageValues > 12, 1, 0)
datos = datos[,c(1:17,19,18)]
#empeora un touch

#2
datos$PageValuesH=NULL

#3
datos$AdministrativeH=NULL

mean(datos$ProductRelated)
q0.75 = quantile(datos$ProductRelated, 0.75)
q0.75
hist(datos$ProductRelated)
datos$ProductRelatedH = ifelse(datos$ProductRelated > 38, 1, 0)
datos = datos[,c(1:17,19,18)]
#mejora medio porciento la tasa de error y lo otro no cambia

#todo esto último analizando una por una, no todas a la vez, y en el benchmark