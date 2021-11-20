setwd("C:/Users/teovo/Google Drive/DI TELLA/8vo semestre/Machine Learning/mlecon")

### Cargamos las librerías:

library(glmnet)
library(ROCR)
library(ggplot2)
library(FNN)
library(class)


rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica

datos = read.table('online_shoppers_intention.CSV',
                   sep=',',header=T,dec='.',na.strings = "N/A")

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

################################################################################
#######################      ordenamos los datos 
################################################################################

set.seed(123)
datos2 <- datos[sample(nrow(datos)),] 

mat = model.matrix(~.-1,datos2)

# Creo 10 'folds'

folds <- cut(seq(1,nrow(mat)),breaks=10,labels=FALSE)

valores.k = c(5,10,25,50,100,200)

TE = c(); # Para cada valor del parámetro computamos el promedio de 10 estimaciones del TEE
TE.sd = c();   # Para cada valor del parámetro computamos el sd del estimador del TEE

for(i in 1:6){ # i recorre los valores del parámetro k.
  TE.kfold = c()
  for(j in 1:10){ # j-recorre los folds
    index.train = which(folds!=j) # j recorre los folds.
    
    train = mat[index.train,]
    test = mat[-index.train,]
    train_y = mat[index.train,70]
    test_y = mat[-index.train,70]
    
    test_pred <- knn(train = train, test = test, cl = as.factor(train_y), k = valores.k[i])
    
    TE.kfold[j] = 1 - sum(diag(table(test_y, test_pred))) / length(test_y)

  }
  
  TE[i] = mean(TE.kfold)
  TE.sd[i] = sd(TE.kfold)
  print(i)  

  }

TE
TE.sd

# Plot de mi primer selección de parámetros por VC (que emoción!) 
plot(valores.k, TE, type='b',pch=20, main='10-fold VC') # :)

valores.k[which(TE==min(TE))]

TE[which(TE==min(TE))]


# ZOOM IN

valores.k = c(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30)

TE = c(); # Para cada valor del parámetro computamos el promedio de 10 estimaciones del TEE
TE.sd = c();   # Para cada valor del parámetro computamos el sd del estimador del TEE

for(i in (1:length(valores.k))){ # i recorre los valores del parámetro k.
  TE.kfold = c()
  for(j in 1:10){ # j-recorre los folds
    index.train = which(folds!=j) # j recorre los folds.
    
    train = mat[index.train,]
    test = mat[-index.train,]
    train_y = mat[index.train,70]
    test_y = mat[-index.train,70]
    
    test_pred <- knn(train = train, test = test, cl = as.factor(train_y), k = valores.k[i])
    
    TE.kfold[j] = 1 - sum(diag(table(test_y, test_pred))) / length(test_y)
    
  }
  
  TE[i] = mean(TE.kfold)
  TE.sd[i] = sd(TE.kfold)
  print(i)  
  
}

# Plot de mi primer selección de parámetros por VC (que emoción!) 
plot(valores.k[-c(1,2)], TE[-c(1,2)], type='b',pch=20, main='10-fold VC') # :)

valores.k[which(TE==min(TE))]

TE[which(TE==min(TE))]

