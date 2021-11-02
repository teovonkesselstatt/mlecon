rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica

library(spData)
data = boston.c

data$TOWN <- NULL
data$TOWNNO <- NULL
data$TRACT <- NULL
data$MEDV <- NULL
data$CHAS <- as.numeric(data$CHAS)

# data <- scale(data) # Si quiero estandarizar de antemano. 
# data <- data.frame(data)

#####################################################
############  RIDGE LASSO Elastic Net:   ############
#####################################################
# install.packages('glmnet')
library(glmnet)
#@@@ ATENCIÓN: Necesitamos codificar como dummy's (one hot) las variables categoricas.
# Nota: glmnet solo adminte datos en formato matriz (no data frames!)
# para trabajar con factores debes codificarlos adecuadamente
# e introducirlos en nuevas columnas de la matriz x
# https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#lin

################################################ 
##### Preparamos los datos antes de modelar ####
################################################
?model.matrix # Expande factores/categoricas a dummies
modmat <- model.matrix( ~ .-1, data); # Creamos la matriz de diseño.
                                      # con el -1 le decimos que no queremos constante.

head(modmat,2)

y = as.matrix(data[,3]) # Variable(s) de respueta(s) en formato 'matrix'

### Creamos IN SAMPLE - OUT OF SAMPLE data sets.
set.seed(1) ; oosid = sample(506,50)
x_oos = modmat[oosid, -c(3)]; 
x_is = modmat[-oosid, -c(3)]; 
y_oos = y[oosid] ; 
y_is = y[-oosid] ;

modmatis <- modmat[-oosid,]

### Creamos TRAIN - TEST datasets
set.seed(1) ; id = sample(456,0.8*456)
x_train = modmatis[id, -c(3)]; x_test = modmatis[-id,-c(3)];
y_train = y[id ] ; y_test = y_is[-id ]

##############################
#### RIDGE (alpha = 0)  ######
##############################

# Algoritmo: min RSS st  "restriccion circular"
# Idea: OLS puede hacer overfitting, entonces restringimos los valores 
# de los parámetros para tener mejores predicciones out of sample 

# Si lambda tiende a 0 --> OLS
# Si lambda tiende a infinito --> predecimos con la oredenada al origen
# Trade off entre sesgo y varianza: elegimos el valor de lambda para minimizar el error 


grid.l =exp(seq(10 , -10 , length = 100)) # El paquete nos permite estimar todo el vector de lambdas

plot(grid.l, type = 'b') 

# Estamos entrenando 100 modelos (1 para cada valor de lambda)
ridge.reg = glmnet(x = x_train , 
                   y = y_train,
                   family = 'gaussian',  # modelo de regresión lineal.
                   alpha = 0,           # Ridge.
                   lambda = grid.l,      # Grilla para lambda.
                   standardize=TRUE)     # por defecto este flag es 'TRUE'. El output luego es desestandarizado!

# Atributos del objeto de regresion

dim(ridge.reg$beta) # Para cada valor de lambda una estimación de los parámetros del modelo
length(ridge.reg$a0) # A cada modelo (equivalentemente para cada lambda) tenemos una estimación de la ordenada al origen. 
coef(ridge.reg)[,1:3] # Parámetros de los primeros 3 modelos

# ¿Cómo cambia cada coeficiente con respecto a lambda?
dev.off()
plot(ridge.reg, xvar = "lambda", label = TRUE, xlab = expression(log(lambda)))

# Predicción: Puedes usar valores de lambda fuera de tu grilla (R se encarga de interpolar).
head(predict(ridge.reg, s = 1 , newx = x_test)) # 's' es el valor de lambda.

######################################################################
###### Aprendiendo el mejor valor de lambda por VC (train-test) ######
######################################################################
ecme.out = c()
for(i in 1:100){
  ridge.pred = predict(ridge.reg, s = grid.l[i] , newx = x_test) # Una sola estimación fuera de la muestra.
  ecme.out[i] = sum((ridge.pred - y_test )^2) / 92
  print(i)
}

# Plot de ECME vs Lambda
dev.off()
plot(log(grid.l), log(ecme.out), type='l', xlab = expression(log(lambda)),
     ylab = expression(log(ecme)), main =  expression(paste('Estimando ', lambda,' optimo' )) )
which(ecme.out==min(ecme.out))

log(grid.l)[42] # Valor Optimo Lambda
abline( v = log(grid.l)[42], col = 'red')

ecme.out[42]    # ECM minimo
points(log(grid.l)[42], log(ecme.out[42]), col = 'blue', pch = 20, lwd = 3)

# Parámetros estimados del modelo asociado a lambda optimo:
coef(ridge.reg)[,42]  # Modelo Optimo

coef(ridge.reg)[,100] # Modelo no regularizado = OLS

coef(ridge.reg)[,42]/coef(ridge.reg)[,100] # Fit relativo a OLS. Notar cambios de signo!

# Chequeando performance en OOS:
ridge.pred = predict(ridge.reg, s = grid.l[42] , newx = x_oos)
ecm.ridge = sum((ridge.pred - y_oos)^2) / 50
log(ecm.ridge) # Log ECM en OOS: -0.30

##########################################################################
######### Aprendiendo el mejor valor de lambda por VC ("5 FOLDS")   ######
##########################################################################
set.seed(1)
cv.out = cv.glmnet(x_is, y_is, lambda = grid.l, alpha = 0, nfolds = 5)
cv.out

# Grafica de ECM como funcion de lambda. En ese caso, nos pone intervalos de ECM para cada lambda y media.
dev.off()           # Verás una gráfica similar solo que tendras para cada valor de lambda
plot(cv.out)        # una estimación de la distribución del ecm (basada en los 5 folds del método de VC).

# Valores "criticos" de lambda
cv.out$lambda.min   # Primera linea desde la izq: lambda que minimiza log(ecm)
cv.out$lambda.1se   # Segunda linea desde la izq: lambda mas grande tal que log(ecm) se mantiene en 1se

bestlam = cv.out$lambda.min

# Una vez que estimamos el mejor valor de lambda (bestlam):
ridge.pred = predict(ridge.reg, s = bestlam , newx = x_oos)
ecm.ridge = sum((ridge.pred - y_oos)^2)/ 50
log(ecm.ridge) # 4.24

##############################
#### LASSO (alpha = 1)   #####
##############################

# Algoritmo: min RSS st  "restriccion cuadrada" 
# Idea: OLS puede hacer overfitting, entonces restringimos los valores 
# de los parámetros para tener mejores predicciones out of sample 

# Si lambda tiende a 0 --> OLS
# Si lambda tiende a infinito --> predecimos con la oredenada al origen
# Trade off entre sesgo y varianza: elegimos el valor de lambda para minimizar el error 

# NOTAR: 1) hace selección de variables (como retriccion lineal, mas soluciones de esquina)
#        2) Los estimadores NO cambian de signo 

grid.l =exp(seq(10 , -10 , length =100))
LASSO.reg = glmnet(x = x_is , 
                   y = y_is,
                   family = 'gaussian', # Regresión. 
                   alpha = 1,           # LASSO 
                   lambda = grid.l,     # Grilla para lambda.
                   standardize=TRUE)    # Flag TRUE por defecto.

# Notar que este modelo "selecciona variables".
dev.off()
plot(LASSO.reg, xvar = "lambda", label = TRUE, xlab = expression(log(lambda)))

## CV para LASSO (alpha = 1)
set.seed(1)
cv.out = cv.glmnet(x_is, y_is, 
                   alpha = 1, 
                   nfolds = 5, 
                   type.measure = c("mse"), # con métrica cross-validamos.
                   lambda = grid.l)         # en que grilla cross-validamos.
cv.out 

plot (cv.out)
bestlam = cv.out$lambda.min
log(bestlam)

LASSO.pred = predict(LASSO.reg, s = bestlam , newx = x_oos)
ecm.lasso = sum((LASSO.pred - y_oos )^2)/50
log(ecm.lasso) # 3.05, mejora considerablemente!

#################################
#### ENet (alpha in (0,1))  #####
#################################

# Promedio entre la distancia de ridge y de lasso
# Antes elegíamos solo lambda
# Ahora también hay que elegir alpha (cómo ponderamos el promedio)


grid.l =exp(seq(10 , -10 , length =100))
ENet.reg = glmnet(x = x_is , 
                  y = y_is,
                  family = 'gaussian', # Regresión. 
                  alpha = 0.5,         # Red elástica con alpha = 1/2 
                  lambda = grid.l,     # Grilla para lambda.
                  standardize=TRUE)    # Flag TRUE por defecto.


### Utilizando las mismas covariables que utilizamos para Ridge y LASSO, 
### compara la performance in--sample y out--of sample del modelo ENet --alpha en (0,1)-- 
################################################# END






# Solución:
ecm.enet = c()
alpha.grid = c(0.1,0.25,0.5,0.75,0.9) # Grilla para el parámetro alpha.
bestlam = c()
for(i in 1:5){
  cv.out = cv.glmnet(x_train, y_train, alpha = alpha.grid[i], nfolds = 10) # para alpha fijo cross valido lambda.
  ENet.reg = glmnet(x = x_train , 
                    y = y_train,
                    family = 'gaussian',             
                    alpha = alpha.grid[i],          # Valor de alpha. 
                    lambda = cv.out$lambda.min)     # Valor optimo de lambda estimado por VC.
  
  bestlam[i] = cv.out$lambda.min
  ENet.pred = predict(ENet.reg, newx = x_test)
  ecm.enet[i] = sum((ENet.pred - y_test )^2) / 92 # computo el valor del ecm para (alpha, lambda^*).
  print(i)
}

plot(alpha.grid, ecm.enet, type = 'b') # De acá selecciono alpha^* Ojo, resultados sensibles a folds!

ENet.reg = glmnet(x = x_is, 
                  y = y_is,
                  family = 'gaussian',             
                  alpha = alpha.grid[4],          # Valor de alpha. 
                  lambda = bestlam[4])     # Valor optimo de lambda estimado por VC.

ENet.pred = predict(ENet.reg, newx = x_oos)
ecm.enet = sum((ENet.pred - y_oos )^2) / 50
log(ecm.enet) # 3.06. Parecido a Lasso.




############################################################################################3 END.
########### Otras opciones para Lasso, ridge y ENets:
#family = 'binomial' --> Modelos Logísticos con regularización
#family = 'poisson' --> Modelos de conteo con regularización
#family = 'mgaussian' --> Modelos de regresión multirespuesta con regularizacion.
############################################### END.


