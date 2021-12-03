################################################################################
#######################     RIDGE y LASSO     ##################################
################################################################################

set.seed(123); id = sample(nrow(datos),0.8*nrow(datos))

Train=datos[id,] # Conjunto de training
Test=datos[-id,]  # Conjunto de test


# Recordemos que la libreria glmnet usa matrices, no data frames
matrix_train = model.matrix( ~ .-1, Train)
matrix_test = model.matrix(~.-1, Test)

set.seed(123); val_id = sample(nrow(matrix_train),0.3*nrow(matrix_train))

matrix_val = matrix_train[val_id,]
matrix_train2 = matrix_train[-val_id,]

x_train <- matrix_train2[,-ncol(matrix_train2)]
y_train = as.matrix(matrix_train2[,ncol(matrix_train2)]) # 0/1 flag.

x_test <- as.matrix(matrix_test[,-ncol(matrix_test)])
y_test<- as.matrix(matrix_test[,ncol(matrix_test)])

x_val <- as.matrix(matrix_val[,-ncol(matrix_val)])
y_val <- as.matrix(matrix_val[,ncol(matrix_val)])

dim(x_train)
dim(y_train)
dim(x_val)
dim(y_val)
dim(x_test)
dim(y_test)

freq_table <- table(y_train) / dim(y_train)[1] 
prob_priori <- as.numeric(freq_table[2])

######################################################
######################  ENETS  #######################
######################################################

alpha = seq(0,1,length.out = 21)
aucs = c() # aca voy guardando los AUCs de cada alpha
  
for (i in 1:length(alpha)){
  grid.l = exp(seq(-10,2,length.out = 25)) # cambiar
  ridge.cv = cv.glmnet(x = x_train, 
                       y = y_train,
                       family = 'binomial', # problema de clasificación
                       alpha = alpha[i], # ridge
                       lambda = grid.l,
                       nfolds = 5,
                       type.measure="auc")
  
  plot (ridge.cv) # La devianza y el auc son métricas que no dependen del umbral (el modelo seleccionado no dependerá del umbral)
  bestlam = ridge.cv$lambda.1se

  ridge.reg = glmnet(x = x_val, 
                     y = y_val,
                     family = 'binomial', # problema de clasificación
                     alpha = alpha[i], # ridge
                     lambda = bestlam)
  
  pred_tr = predict(ridge.reg, s = bestlam, newx = x_train, type = 'response')
  pred_train <- prediction(pred_tr, y_train)
  perf_train <- performance(pred_train,"tpr","fpr")
  auc <- performance(pred_train, measure = "auc")
  aucs[i] = auc@y.values[[1]]
  print(i)
}

plot(aucs)
auc.opt = aucs[which(aucs==max(aucs))]
abline(v = which(aucs==max(aucs)) , col = 'red', lty = 2 )

alpha.opt = alpha[which(aucs==max(aucs))]

alpha.opt

############################# Mejor Modelo: ####################################

set.seed(123)
ridge.cv = cv.glmnet(x = x_train, 
                     y = y_train,
                     family = 'binomial', # problema de clasificación
                     alpha = alpha.opt, # ridge
                     lambda = grid.l,
                     nfolds = 5,
                     type.measure="auc")

bestlam = ridge.cv$lambda.1se

ridge.reg = glmnet(x = x_val, 
                   y = y_val,
                   family = 'binomial', # problema de clasificación
                   alpha = alpha.opt, # ridge
                   lambda = bestlam)


####################### Performance en el TEST: ################################

pred_test = predict(ridge.reg, s = bestlam, newx = x_test, type = 'response')

pred2 <- prediction(pred_test, y_test)
perf2 <- performance(pred2,"tpr","fpr")

# Graficamos la curva ROC
plot(perf2, main="Curva ROC en el conjunto de TEST", colorize=T)

auc_test <- performance(pred2, measure = "auc")
auc_test@y.values[[1]] # 0.8885 IMPORTANTE


######################## ELIGIENDO EL THRESHOLD ################################

# Hago Validation Set Approach para elegir el que maximice F2 score

# Predicciones sobre datos de VALIDACION
pred_val = predict(ridge.reg, s = bestlam , newx = x_val, type = 'response')

thresh = seq(0.05,0.5,length.out = 200)

# Optimizamos el threshold eligiendo el que maximiza F2.
F2.est = c();
beta = 2

for(i in 1:length(thresh)){
  y_hat = as.numeric(pred_val >= thresh[i])
  TP = sum( y_hat*y_val==1 )     # True +
  FP = sum( y_hat*(1-y_val)==1 ) # False +
  FN = sum( (1-y_hat)*y_val==1 ) # False -
  prec = TP/(TP + FP) ; rec = TP/(TP + FN)
  F2.est[i] = (1+beta^2)*(prec*rec)/((beta^2)*prec + rec)
}

plot(thresh, F2.est, type = 'b', bty = 'n')
thresh.opt = thresh[which(F2.est==max(F2.est))]
abline(v = thresh.opt , col = 'red', lty = 2 )

# Diferencia de Punto de Corte Optimo vs Prob a Priori
thresh.opt;  prob_priori # 0.1291 y 0.1523

# Performance en Test:
pred_test = predict(ridge.reg, s = bestlam , newx = x_test, type = 'response')
y_hat = as.numeric(pred_test>= thresh.opt)

TP = sum( y_hat*y_test==1 )     # True +
FP = sum( y_hat*(1-y_test)==1 ) # False +
FN = sum( (1-y_hat)*y_test==1 ) # False -
prec = TP/(TP + FP) ; rec = TP/(TP + FN)

F2.est = (1+beta^2)*(prec*rec)/((beta^2)*prec + rec)

matriz.confusion = table(y_hat, y_test)
matriz.confusion

tasa_error_test = 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error_test # 15.17% USANDO THRESH.OPT. 
F2.est # 70.85 DE F2 score!

# Lo que estamos haciendo es lo siguiente:
# 1. Separamos los datos entre train, validation y test. 
# 2. usando los datos de train, encontramos el lambda optimo dado cada alpha. 
# Con ese lambda, usamos los datos de validacion para elegir el alpha que mejor AUC tiene fuera del train.

# Con eso tenemos el alpha y el lambda optimos.
# Luego, usamos de vuelta el validation set para elegir el treshold que maximiza el el F2 score.
# Con el treshold optimo, medimos el F2 score y la tasa de error sobre el conjunto de test.













