################################################################################
#################     Benchmark: Regresion Logistica    ########################
################################################################################
rm(list=ls()) # Limpiamos la memoria.
dev.off()     # Limpiamos la ventana gráfica

datos = read.table('datos.csv',
                   sep=',',header=T,dec='.',na.strings = "N/A")

quiero.ver.con.weights = FALSE

if (!quiero.ver.con.weights){
  datos$weights = 1
}

datos$weights[which.min(datos$weights)]    

datos$Month <- factor(datos$Month, levels = c("Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), ordered = TRUE)
datos$OperatingSystems <- factor(datos$OperatingSystems)
datos$Browser <- factor(datos$Browser)
datos$Region <- factor(datos$Region)
datos$TrafficType <- factor(datos$TrafficType)
datos$VisitorType <- factor(datos$VisitorType)
datos$Revenue <- factor(datos$Revenue)
datos$Weekend <- factor(datos$Weekend)

######### Separo en TRAIN, TEST Y VALIDATION ###################################


set.seed(123); id = sample(nrow(datos),0.8*nrow(datos))

Train=datos[id,] # Conjunto de training
Test=datos[-id,]  # Conjunto de test


# Paso los df a matrices
matrix_train = model.matrix( ~ .-1, Train)
matrix_test = model.matrix(~.-1, Test)

# Hago un set de Validacion para hacer Validation Set Approach
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
######### REGRESION LOGISTICA  #######################
######################################################

logit.reg = glmnet(x = x_train[,-ncol(x_train)], 
                   y = y_train,
                   family = 'binomial', # problema de clasificación
                   alpha = 1, 
                   lambda = 0,
                   weights = x_train[,ncol(x_train)]) # sin regularizacion (reg logistica normal)


pred_tr = predict(logit.reg, s = 0, newx = x_train[,-ncol(x_train)], type = 'response')


####################### Performance en el TRAIN: ###############################

# El umbral de corte que vamos a usar para decidir sobre una nueva observacion
# si prob>prob_priori --> clasificamos como 1

pred_train <- prediction(pred_tr, y_train)
perf_train <- performance(pred_train,"tpr","fpr")

# Graficamos la curva ROC
plot(perf_train, main="Curva ROC en el conjunto de TRAIN", colorize=T)

auc <- performance(pred_train, measure = "auc")
auc@y.values[[1]] # 0.9045, pero no me importa esto. AHORA: 0.8297

####################### Performance en el TEST: ################################

# Predicciones sobre datos TEST 
pred_test = predict(logit.reg, s = 0 , newx = x_test[, -ncol(x_test)], type = 'response')

pred2 <- prediction(pred_test, y_test)
perf2 <- performance(pred2,"tpr","fpr")

# Graficamos la curva ROC
plot(perf2, main="Curva ROC en el conjunto de TEST", colorize=T)

auc_test <- performance(pred2, measure = "auc")
auc_test@y.values[[1]] # 0.8671 MUY IMPORTANTE. AHORA: 0.7963


######################## ELIGIENDO EL THRESHOLD ################################

# Hago Validation Set Approach

# Predicciones sobre datos de VALIDACION
pred_val = predict(logit.reg, s = 0 , newx = x_val[,-ncol(x_val)], type = 'response')

thresh = seq(0.05,0.5,length.out = 200)

# Optimizamos el threshold eligiendo el que maximiza F1.
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

x11();plot(thresh, F2.est, type = 'b', bty = 'n')
thresh.opt = thresh[which(F2.est==max(F2.est))]
abline(v = thresh.opt , col = 'red', lty = 2 )

# Diferencia de Punto de Corte Optimo vs Prob a Priori
thresh.opt # 0.1337. AHORA: 0.1043
prob_priori # 0.1524

# Performance en Test:
pred_test = predict(logit.reg, s = 0 , newx = x_test[,-ncol(x_test)], type = 'response')
y_hat = as.numeric(pred_test>= thresh.opt)

TP = sum( y_hat*y_test==1 )     # True +
FP = sum( y_hat*(1-y_test)==1 ) # False +
FN = sum( (1-y_hat)*y_test==1 ) # False -
prec = TP/(TP + FP) ; rec = TP/(TP + FN)

F2.est = (1+beta^2)*(prec*rec)/((beta^2)*prec + rec)

matriz.confusion = table(y_hat, y_test)
matriz.confusion

tasa_error_test = 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error_test # 21.94% USANDO THRESH.OPT. AHORA: 21.65%
F2.est # 65.13 DE F2 score! AHORA: 0.5782
auc_test@y.values[[1]] # 0.8671 MUY IMPORTANTE. AHORA: 0.7963

x11();plot(sort(pred_test), ylab = 'Valor Predecido', xlab = 'Observación')
abline(h = thresh.opt, col = "red")
abline(h = prob_priori, col = "blue")
