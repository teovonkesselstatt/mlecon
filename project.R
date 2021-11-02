setwd("C:/Users/teovo/Google Drive/DI TELLA/8vo semestre/Machine Learning/mlecon")

### Cargamos las librerías:

require(glmnet)|| install.packages('glmnet')  #regularización de modelos lineales
require(ROCR)  || install.packages('ROCR')
require(ggplot2) || install.packages('ggplot2')
library(glmnet)
library(ROCR)
library(ggplot2)
require(FNN) || install.packages('FNN')

rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica

datos = read.table('online_shoppers_intention.CSV',
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
table(datos$Revenue)

### VARIABLES CATEGÓRICAS
g1 <- ggplot(data=datos, aes(Month))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!

g1

g2 <- ggplot(data=datos, aes(VisitorType))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!

g2

g3 <- ggplot(data=datos, aes(Weekend))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!

g3

g4 <- ggplot(data=datos, aes(OperatingSystems))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!

g4

g5 <- ggplot(data=datos, aes(Browser))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!

g5

g6 <- ggplot(data=datos, aes(Region))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!

g6

g7 <- ggplot(data=datos, aes(TrafficType))+
  geom_bar(aes(fill=Revenue), position="fill") # podemos graficar por categorias!

g7

# Analisis de variables numericas

g8 <- ggplot(data=datos, aes(SpecialDay, color = Revenue, fill = Revenue))+
  geom_density(alpha = 0.1)
g8

g9 <- ggplot(data=datos, aes(ProductRelated_Duration, color = Revenue, fill = Revenue))+
  geom_density(alpha = 0.1)
g9

