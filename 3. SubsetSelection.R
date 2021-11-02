rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica

##################################################################
# Boston Median Housing Prices: Predecir valor mediano de casas  #
##################################################################
require(spData) || install.packages(spData)
library(spData)
?boston.c
data = boston.c 

data$TOWN <- NULL
data$TOWNNO <- NULL
data$TRACT <- NULL
data$MEDV <- NULL

n = dim(data)[1]

set.seed(1) ; id = sample(n,0.8*n)

train = data[ id, ] # Datos de entrenamiento. 
test  = data[-id, ] # Datos de validacion.

#########################
# Best subset selection #
#########################
require(leaps)|| install.packages('leaps') 
library(leaps)

##################################
### PRIMERO: Método exhaustivo ###
##################################

### Algoritmo: se fija para cada "tamaño" de modelo cuáles son las covariables a incluir. 
### Es decir, para i=1 prueba todas las opciones de y~x_j y elige la x_j que 
### major R^2 tiene. (hay tantos modelos como cantidad de covariables)
### Para i=2 prueba todas las combinaciones y~x_j+x_k y elige el mejor. 
### (hay p!/(2!(p-2)!) formas posibles)
### Para cada tamaño de modelo, elige el mejor. Luego, entre todos esos candidatos 
### elige el que maximiza el R^2


# Busqueda exhaustiva de modelo (factible cuando p < 30)
regfit.full = regsubsets(CMEDV ~ . ,    # R hace One-hot encoding automáticamente (¡ojo con el formato de las covariables!).
                         data = train, 
                         method = 'exhaustive', # Método de selección.
                         force.in = NULL,   # Si quiero forzar a que una covariable aparezca.
                         force.out = NULL,  # Si quiero forzar a que una covariable NO aparezca.
                         intercept = TRUE,  # flag intercepto.
                         nvmax = 15,      # Número máximo de covariables a considerar (<= p) (Nota: No es necesario que utilices esta opción si p <= 30)
                         nbest = 1)       # Cuantos best quiero guardar para cada tamaño de modelo.

### Estimaciones in-sample:
summary(regfit.full) # Lista de los mejores modelos por tamaño de modelo (M_0,...,M_15) 

summary(regfit.full)$adjr2  # R cuadrados ajustados in sample. Me permite comparar modelos de diferente tamaño
# Se recomienda igualmente estimar el R2a fuera de la muestra.

# ¿Qué modelo tiene el R2 ajustado más grande? (in sample)
which(summary(regfit.full)$adjr2 == max(summary(regfit.full)$adjr2)) 

# El modelo con las 12 covariables siguientes: 
coef(regfit.full,12)              # Variables incluidas
summary(regfit.full)$adjr2[12]    # R2a mas alto

# Comparacion grafica
dev.off()
plot(1:15, summary(regfit.full)$adjr2,xlab="# variables",
     ylab="R2a",type="b",pch=20)
points(12 , summary(regfit.full)$adjr2[12] , col =" red " , cex = 2 , pch = 20)


#####################################################
############    Forward & Backward     ############## (factible aún cuando p > 30)
#####################################################


### FORWARD
### Algoritmo: similar al exhaustivo pero menos costo computacional. 
### Empieza estimando un modelo con 1 covariable. Elige el que maximiza el R^2. 
### Para los modelos de tamaño 2 usa lo que aprendió en la etapa anterior. 
### Entonces no estima todas las posibles combinaciones de covariables, sino 
### que va a agregando features de a uno, notando cuál fue el mejor la etapa anterior. 

regfit.fwd = regsubsets(CMEDV ~ . ,
                        data = train, 
                        method = 'forward',  # Stepwise method.
                        force.in=NULL,       # Si quiero forzar a que una covariable aparezca.
                        force.out=NULL,      # Si quiero forzar a que una covariable NO aparezca.
                        intercept=TRUE,      # flag intercepto.
                        nvmax=15,            
                        nbest = 1)
summary(regfit.fwd) # Resumen de la secuencia M_0,...,M_15.

which(summary(regfit.fwd)$adjr2 == max(summary(regfit.fwd)$adjr2) ) # In sample!
summary(regfit.fwd)$which[12,]
summary(regfit.fwd)$adjr2[12]


### BACKWARD
### Algoritmo: empieza con el modelo "saturado" con todas las covariables. 
### Ahora en vez de ir agregando, vamos sacando covariables de a una.
### En cada momento se fija todos los modelos que tienen i-1 features. 
###Y elige el que mayor R^2.

regfit.bkw = regsubsets(CMEDV ~ . ,
                        data = train, 
                        method = 'backward', 
                        nvmax=15, 
                        nbest = 1)
summary(regfit.bkw)  # Resumen de la secuencia M_0,...,M_15.

which(summary(regfit.bkw)$adjr2 == max(summary(regfit.bkw)$adjr2) ) # In sample!
summary(regfit.bkw)$which[12,] 
summary(regfit.bkw)$adjr2[12]





####################
# CROSS VALIDATION #
####################

# Recordar que hay que utilizar alguna medida sensible a tamaño del modelo.

### TRAIN - TEST CV para el caso EXHAUSTIVO ###

test.R2a = c() # R2 ajustado.
form = as.formula(regfit.full$call[[2]])
mat = model.matrix(form, test) # Matriz de diseño (conjunto de test) que necesito para poder hacer "y.hat = X*beta" 

for(i in 1:15){
  coefi = coef(regfit.full, id=i) #Generamos una matriz con los betas de cada regresion
  y.hat = mat[, names(coefi)]%*%coefi #y=X*beta
  R = 1- sum( (y.hat - test$CMEDV)^2)/sum((test$CMEDV - mean(test$CMEDV))^2) # R2 = 1 - RSS/TSS
  test.R2a[i] = 1 - (dim(mat)[1] - 1)/ (dim(mat)[1] - dim(mat)[2])* (1-R) # Fórmula de R2a  
}

which(test.R2a == max(test.R2a))
coef(regfit.full, id=11) # Mejor modelo validando contra la muestra de test.
# cmedv ~ crim + zn + chas1 + nox + rm + dis + rad + tax + ptratio + black + lstat





### k-Fold cross validation ###

set.seed(123)
folds=sample(rep(1:5, length=nrow(train)))
folds; table(folds) # parto la muestra de train en 5 folds (mismo tamaño).

# Creamos funcion para que prediga con cada fold. 
predict.regsubsets=function(object, newdata, id, ...){ #id le decimos a R de q tamaño de modelo se trata.
  form=as.formula(object$call[[2]]) # formula general del modelo (object) a utilizar para predecir
  mat=model.matrix(form, newdata) # llevar a forma matricial en conjunto con data de prediccion (newdata)
  coefi=coef(object, id=id) # (id = tamaño del modelo)
  mat[, names(coefi)]%*%coefi
}

cv.R2a=matrix(NA, 5, 15) # folds por filas, modelos $M_1,...M_15$ por columnas.
for (i in 1:5){ #i-recorre folds.
  best.fits = regsubsets(CMEDV ~ . ,
                         data = train[folds!=i,], # Quitamos el fold i para estimar $M_1,...M_15$
                         method = 'exhaustive', 
                         nvmax = 15, nbest = 1)
  for (j in 1:15){ # j-recorre covariables.
    y.hat = predict.regsubsets(best.fits, train[folds==i,], id=j) # Con el fold i estimamos el ECM utilizando $M_1,...M_11$ (j)
    R = 1- sum((y.hat - train[folds==i,3])^2)/sum((train[folds==i,3] - mean(train[folds==i,3]))^2) # R2 = 1 - RSS/TSS
    cv.R2a[i,j] = 1 - (dim(mat)[1] - 1)/ (dim(mat)[1] - dim(mat)[2])* (1-R) # Fórmula de R2a  
  }
}
cv.R2a # 5 folds por 15 tamaños de modelo.
est.R2a = apply(cv.R2a, 2, mean)     # Estimación del R2a por validación cruzada.
est.R2a
se = sqrt(apply(cv.R2a, 2, var)/5)   # Errores estandar.
se

which(est.R2a == max(est.R2a))  #15
coef(regfit.full,15)     # Mejor modelo in-sample

dev.off()
plot(1:15,est.R2a, pch=19, type="b", main ='MSE estimado por VC', xlab = '# de covariables')

#### Una vez seleccionado el modelo hacemos predicciones:  
modelo.final = lm(CMEDV ~ ., data = train) 
summary(modelo.final)

pred.test = predict(modelo.final, newdata = test)
ecm.out.sample = sum((test$CMEDV - pred.test)^2)/102
log(ecm.out.sample) 





