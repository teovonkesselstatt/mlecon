##########################################
# Machine Learning para Economistas 2021 #
######### Gráficos con ggplot2 ###########
##########################################
rm(list=ls())
dev.off()
#install.packages('ggplot2')
library(ggplot2)

# Abrimos la base de datos
bikes = read.csv("bikes.csv", header=T)
dim(bikes)
head(bikes)

attach(bikes)


reglin <- lm(cnt~temp, data=bikes)
plot(reglin)


#Para realizar el gráfico "usual" de regresión lineal usamos ggplot: 
#Se va agregando al gráfico por "capas":
ggplot(bikes, aes(x=temp, y=cnt)) #no grafica nada

## Scatterplot:
ggplot(bikes, aes(x=temp, y=cnt))+geom_point()

## Scatterplot + recta de regresión:
ggplot(bikes, aes(x=temp, y=cnt))+geom_point()+geom_smooth(method='lm',se=F)

## Podemos cambiar el tema, agregar títulos y nombres a los ejes:
#install.packages('ggthemes')
library(ggthemes)
ggplot(bikes, aes(x=temp, y=cnt))+geom_point()+geom_smooth(method='lm', se=F)+theme_classic()

ggplot(bikes, aes(x=temp, y=cnt))+geom_point()+geom_smooth(method='lm', se=F)+theme_stata()+
  ggtitle("Regresión lineal")+xlab("Temperatura")+ylab("Cantidad")

## Histogramas:
ggplot(bikes, aes(x=cnt))+geom_histogram(binwidth=200, color="black", fill="lightblue")+
  geom_vline(aes(xintercept=mean(cnt)),
             color="dark blue", linetype="dashed", size=1)

##Gráficos según categoría: 
bikes$mnth <- as.factor(bikes$mnth)

ggplot(bikes, aes(x=temp, y=hum, col=mnth))+geom_point()







#Para ejemplificarlo podemos generar dos series de datos: 
norm1 <- as.matrix(rnorm(100, 0, 1)) 
norm2 <- as.matrix(rnorm(100, 1, 1)) 

#Las apilamos
normales<-rbind(norm1, norm2)

#Generamos una nueva variable. Vale 1 si viene de la normal con media 0.
grupo <- as.matrix(rep(c(0, 1), each = 100))

#Las pegamos:
datos<- cbind(normales, grupo)
datos_df <- as.data.frame(datos)
datos_df$V2 <- as.factor(datos_df$V2)

#Graficamos un Histograma y la densidad según categoría: 
ggplot(datos_df, aes(V1, fill=V2))+geom_histogram(position = "dodge", binwidth = 0.5)

ggplot(datos_df, aes(V1, fill=V2))+geom_density(position = "dodge", binwidth = 0.5, alpha=0.5)




