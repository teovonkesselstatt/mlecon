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

#### Diccionario de variables:
# "Administrative": number of different administrative pages visited in that session
# "Administrative Duration": total time spent on those pages
# "Informational": number of different informational pages visited in that session
# "Informational Duration": total time spent on those pages
# "Product Related": number of different product related pages visited in that session
# "Product Related Duration": total time spent on those pages

# "Bounce Rate": percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session.
# "Exit Rate":  the percentage of pageviews that were the last in the session.
# "Page Value": average value for a web page that a user visited before completing an e-commerce transaction.

# "Special Day": indicates the closeness of the site visiting time to a specific special day
# "Month": No hay enero ni abril
# "Operating Systems"
# "Browser"
# "Region"
# "Traffic Type": ??

# "Visitor type": ??
# "Weekend": Boolean for weekend or not
# "Revenue": If the person bought or not

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


# Vemos cuántas observaciones de cada categoría hay:
]

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
require(corrplot) || install.packages('corrplot')  
library(corrplot)

numeric.var <- sapply(datos$Administrative, numeric.var)
corr.matrix <- cor(datos$Administrative[,numeric.var])
corrplot(corr.matrix, main="\nCorrelation Plot for Numerical Variables", method="number")

############ Sacando Outliers ###########

datos$log = log(datos$Administrative_Duration + 1)
boxplot(datos$log)

boxplot(datos$Revenue)
plot(datos$Weekend,datos$Revenue)

plot(datos$OperatingSystems,datos$Administrative_Duration)

datos$administrative1 <- datos[datos$Administrative ==0,]
  
boxplot(datos$OperatingSystems if datos$OperatingSystems>0)

remove = c(0)
hola <- subset(datos, !(datos$Administrative_Duration %in% remove))
hola$log = log(hola$Administrative_Duration)
boxplot(hola$log)

#Reescalando los outliers superiores#
q0.99 = quantile(datos$Administrative_Duration, 0.99) 
datos$Administrative_Duration[which(datos$Administrative_Duration>q0.99)] = q0.99

$Volviendo a graficar para chequear que nos da bien la reescalación$
hola <- subset(datos, !(datos$Administrative_Duration %in% remove))
hola$log = log(hola$Administrative_Duration)
boxplot(hola$log)

boxplot(datos$Administrative)

q0.99 = quantile(datos$Administrative, 0.99) 
datos$Administrative[which(datos$Administrative>q0.99)] = q0.99

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

##############Ploteo Conjunto de Variables##################

plot(datos$ProductRelated_Duration,datos$ProductRelated, col=datos$Revenue)
plot(datos$SpecialDay,datos$ProductRelated_Duration, col=datos$Revenue)
plot(datos$Month,datos$SpecialDay, col=datos$Revenue)
plot(datos$Weekend,datos$SpecialDay, col=datos$Revenue)

plot(datos$Administrative_Duration,datos$Month,col=datos$Revenue)
plot(datos$OperatingSystems,datos$Administrative_Duration,col=datos$Revenue)
plot(datos$Browser,datos$Administrative_Duration,col=datos$Revenue)

plot(datos$Informational_Duration,datos$Administrative_Duration,col=datos$Revenue)
plot(datos$ProductRelated_Duration,datos$Administrative_Duration,col=datos$Revenue)

############## 3/12/21 ##############

#viendo un poco más distribución de variables.
#ni ahi normalidad.
hist(datos$Administrative)
hist(datos$Administrative_Duration)
hist(datos$ProductRelated)
hist(datos$ProductRelated_Duration)
hist(datos$SpecialDay)

# Regresi?n robusta: (copiada de Gabriel)
summary(r.rob <- rlm(crime ~ poverty + single, data = cdata))
# El coef asociado a poverty (no significativo al 5%) ahora es positivo. 

w <- data.frame(state = cdata$state, resid = r.rob$resid, weight = r.rob$w)
w[c(1,2,3,9, 25, 51),]
# --------------------------------------------------------------End.