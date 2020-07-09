ratingProbsFit <- function(dataIn,maxRating,predMethods,embedMeans,specialArgs,newData){
  max <- as.numeric(maxRating)
  method <- as.character(predMethods)
  x <- dataIn
  names(x)[1] <- "V1"
  names(x)[2] <- "V2"
  names(x)[3] <- "V3"
  if(method == "logit")
  {
    if(embedMeans == TRUE)
    {
      col_1_means <- tapply(x$V3,x$V1,mean)
      col_2_means <- tapply(x$V3,x$V2,mean)
      x_emb <- x
      x_emb$V1 <- col_1_means[x$V2]
      x_emb$V2 <- col_2_means[x$V2]
      x_emb$V1 <- as.vector(x_emb$V1)
      x_emb$V2 <- col_2_means[x$V2]
      x_emb$V2 <- as.vector(x_emb$V2)
      x_emb$V3 <- as.factor(x_emb$V3)
      range_1 <- as.integer(x_emb$V3 == 1)
      range_2 <- as.integer(x_emb$V3 == 2)
      range_3 <- as.integer(x_emb$V3 == 3)
      range_4 <- as.integer(x_emb$V3 == 4)
      range_5 <- as.integer(x_emb$V3 == 5)
      glmout_1 <- glm(range_1 ~ V1 + V2, data = x_emb, family = "binomial")
      glmout_2 <- glm(range_2 ~ V1 + V2, data = x_emb, family = "binomial")
      glmout_3 <- glm(range_3 ~ V1 + V2, data = x_emb, family = "binomial")
      glmout_4 <- glm(range_4 ~ V1 + V2, data = x_emb, family = "binomial")
      glmout_5 <- glm(range_5 ~ V1 + V2, data = x_emb, family = "binomial")
      probsFitOut <- list(method = 1, data_1 = glmout_1, data_2 = glmout_2, data_3 = glmout_3, data_4 = glmout_4, data_5 = glmout_5) 
      class(probsFitOut) <- "recProbs"
      predict(probsFitOut, newData)
    } else {
    range_1 <- as.integer(x$V3 == 1)
    range_2 <- as.integer(x$V3 == 2)
    range_3 <- as.integer(x$V3 == 3)
    range_4 <- as.integer(x$V3 == 4)
    range_5 <- as.integer(x$V3 == 5)
    glmout_1 <- glm(range_1 ~ V1 + V2, data = x, family = "binomial")
    glmout_2 <- glm(range_2 ~ V1 + V2, data = x, family = "binomial")
    glmout_3 <- glm(range_3 ~ V1 + V2, data = x, family = "binomial")
    glmout_4 <- glm(range_4 ~ V1 + V2, data = x, family = "binomial")
    glmout_5 <- glm(range_5 ~ V1 + V2, data = x, family = "binomial")
    probsFitOut <- list(method = 1, data_1 = glmout_1, data_2 = glmout_2, data_3 = glmout_3, data_4 = glmout_4, data_5 = glmout_5) 
    class(probsFitOut) <- "recProbs"
    predict(probsFitOut, newData)
    }
  }
    
  if(method == "NMF")
  {
    # songsDataset <- read.csv("songsDataset.csv")
    # x<- songsDataset
    rank <- as.numeric(specialArgs)
    r <- Reco()
    #x <- cbind(InstEval[,c(1:2,7)])
    x<-dataIn
    col1 <- data.frame()
    col2 <- data.frame()
    col3 <- data.frame()
    col4 <- data.frame()
    col5 <- data.frame()
    final <- data.frame()
    number <- nrow(x)
    #number <- 10000
    row <- 1 : number
    #print("working1")
    for (i in row)
    {
      if (x[i,3] ==1)
      {
        col1 <- rbind(col1, c(1))
        col2 <- rbind(col2, c(0))
        col3 <- rbind(col3, c(0))
        col4 <- rbind(col4, c(0))
        col5 <- rbind(col5, c(0))
      }
      if (x[i,3] ==2)
      {
        col1 <- rbind(col1, c(0))
        col2 <- rbind(col2, c(1))
        col3 <- rbind(col3, c(0))
        col4 <- rbind(col4, c(0))
        col5 <- rbind(col5, c(0))
      }
      if (x[i,3] ==3)
      {
        col1 <- rbind(col1, c(0))
        col2 <- rbind(col2, c(0))
        col3 <- rbind(col3, c(1))
        col4 <- rbind(col4, c(0))
        col5 <- rbind(col5, c(0))
      }
      if (x[i,3] ==4)
      {
        col1 <- rbind(col1, c(0))
        col2 <- rbind(col2, c(0))
        col3 <- rbind(col3, c(0))
        col4 <- rbind(col4, c(1))
        col5 <- rbind(col5, c(0))
      }
      if (x[i,3] ==5)
      {
        col1 <- rbind(col1, c(0))
        col2 <- rbind(col2, c(0))
        col3 <- rbind(col3, c(0))
        col4 <- rbind(col4, c(0))
        col5 <- rbind(col5, c(1))
      }
    }
    #print("working2")
    first <- x[1:number,1]
    second <- x[1:number, 2]
    data <- data.frame(first, second, col1, col2, col3, col4, col5)
    head(data)
    #--------------------------------------------------------------------------------
    #Running NMF
    
    ie3.trn <- data_memory(data[,1], data[,2], data[,3], index1= TRUE)
    r$train(ie3.trn, opts= list(dim=rank, nmf=TRUE))
    result1 <- r$output(out_memory(), out_memory())
    w1 <- result1$P
    h1 <- t(result1$Q)
    #-------------
    ie4.trn <- data_memory(data[,1], data[,2], data[,4], index1= TRUE)
    r$train(ie4.trn, opts= list(dim=rank, nmf=TRUE))
    result2 <- r$output(out_memory(), out_memory())
    w2 <- result2$P
    h2 <- t(result2$Q)
    #-------------
    ie5.trn <- data_memory(data[,1], data[,2], data[,5], index1= TRUE)
    r$train(ie5.trn, opts= list(dim=rank, nmf=TRUE))
    result3 <- r$output(out_memory(), out_memory())
    w3 <- result3$P
    h3 <- t(result3$Q)
    #--------------
    ie6.trn <- data_memory(data[,1], data[,2], data[,6], index1= TRUE)
    r$train(ie6.trn, opts= list(dim=rank, nmf=TRUE))
    result4 <- r$output(out_memory(), out_memory())
    w4 <- result4$P
    h4 <- t(result4$Q)
    #---------------
    ie7.trn <- data_memory(data[,1], data[,2], data[,7], index1= TRUE)
    r$train(ie7.trn, opts= list(dim=rank, nmf=TRUE))
    result5 <- r$output(out_memory(), out_memory())
    w5 <- result5$P
    h5 <- t(result5$Q)
    #print("working3")
    
    probsFitOut <- list(method = 2,  data_1=w1, data_2=h1, data_3=w2, data_4=h2, data_5 = w3, data_6=h3, data_7=w4, data_8=h4, data_9=w5, data_10=h5) # <---- Everything for NMF goes here  
    class(probsFitOut) <- "recProbs"
    predict(probsFitOut,newData )
    
  }
  if(method == "kNN")
  {
    
    #origData <- data.frame(InstEval[,c(1:2,7)])
    origData <- dataIn
    names(origData) <- c("userID", "itemID", "rating")
    origData$rating <- as.factor(origData$rating)
    ratToDummies <- factorToDummies(origData$rating, "rating")
    origData <- origData[, -3]
    origData <- cbind(origData, ratToDummies)
    inpData<- list(rep(NA, maxRating), type =any)
    for(i in 1:(maxRating-1)){
      inpData[[i]] <- formUserData(origData[, c(1:2, i+2)])
    }
    probsFitOut <- list(method = 3, data_1 = maxRating, data_2= inpData, data_3=specialArgs) # <---- Everything for KNN goes here 
    class(probsFitOut) <- "recProbs"
    # print("working")
    predict(probsFitOut, newData)
    
    
  }
  if(method == "CART")
  {
    col_1_means <- tapply(x$V3,x$V1,mean)
    col_2_means <- tapply(x$V3,x$V2,mean)
    x_emb <- x
    x_emb$V1 <- col_1_means[x$V2]
    x_emb$V2 <- col_2_means[x$V2]
    x_emb$V1 <- as.vector(x_emb$V1)
    x_emb$V2 <- col_2_means[x$V2]
    x_emb$V2 <- as.vector(x_emb$V2)
    x_emb$V3 <- as.factor(x_emb$V3)
    #ctout <- ctree(V3 ~ V1+V2,data=x_emb)
    ctout <- ctree(V3 ~ V1+V2,data=x_emb, control = ctree_control(minsplit = 10))
    #ctout <- ctree(V3 ~ V1+V2,data=x_emb, control = ctree_control(minsplit = 20))
    #ctout <- ctree(V3 ~ V1+V2,data=x_emb, control = ctree_control(minsplit = 5))
    #ctout <- ctree(V3 ~ V1+V2,data=x_emb, control = ctree_control(maxdepth = 3))
    #ctout <- ctree(V3 ~ V1+V2,data=x_emb, control = ctree_control(maxdepth = 5))
    #plot(ctout, type = "simple")
    probsFitOut <- list(data = ctout, method = 4)
    class(probsFitOut) <- "recProbs"
    predict(probsFitOut, newData)
    #head(output)
    #error_table <- table(actual = x$V3, fitted = output)
    #print(error_table)
    #misclassification_error <- (1 - (sum(diag(error_table))/sum(error_table)))
    #print(misclassification_error)
    
  }
  
}

predict.recProbs <- function(probsFitOut, newXs){
  if(probsFitOut$method == 1)
  {
    predicted_ratings <- data.frame(prob_1 = exp(predict(probsFitOut$data_1, newXs)), prob_2 = exp(predict(probsFitOut$data_2, newXs)), prob_3 = exp(predict(probsFitOut$data_3, newXs)), prob_4 = exp(predict(probsFitOut$data_4, newXs)), prob_5 = exp(predict(probsFitOut$data_5, newXs)))
    rows <- 1:nrow(predicted_ratings)
    for (i in rows)
    {
      sum <- predicted_ratings[i,1] + predicted_ratings[i,2] + predicted_ratings[i,3] + predicted_ratings[i,4] + predicted_ratings[i,5]
      scale <- 1/sum
      predicted_ratings[i,1] <- predicted_ratings[i,1]*scale
      predicted_ratings[i,2] <- predicted_ratings[i,2]*scale
      predicted_ratings[i,3] <- predicted_ratings[i,3]*scale
      predicted_ratings[i,4] <- predicted_ratings[i,4]*scale
      predicted_ratings[i,5] <- predicted_ratings[i,5]*scale
      
    }
    print(predicted_ratings)
  }
  if(probsFitOut$method == 2)
  {
    print("we are predicting NMF")
    w1 <- probsFitOut$data_1
    h1 <- probsFitOut$data_2
    w2 <- probsFitOut$data_3
    h2 <- probsFitOut$data_4
    w3 <- probsFitOut$data_5
    h3 <- probsFitOut$data_6
    w4 <- probsFitOut$data_7
    h4 <- probsFitOut$data_8
    w5 <- probsFitOut$data_9
    h5 <- probsFitOut$data_10
    
    #------------------------------
    size <- nrow(newXs)
    rat1 <- vector(length=size)
    rat2 <- vector(length=size)
    rat3 <- vector(length=size)
    rat4 <- vector(length=size)
    rat5 <- vector(length=size)
    for (i in 1:size)
    {
      j<- newXs[i,1]
      k<- newXs[i,2]
      if(is.na(j) || is.na(k))
      {
        rat1[i]<- NA
        rat2[i]<- NA
        rat3[i]<- NA
        rat4[i]<- NA
        rat5[i]<- NA
      }
      else
      {
        rat1[i] <- w1[j,] %*%  h1[,k]
        rat2[i] <- w2[j,] %*% h2[,k]
        rat3[i] <- w3[j,] %*% h3[,k]
        rat4[i] <- w4[j,] %*% h4[,k]
        rat5[i] <- w5[j,] %*% h5[,k]
      }
    }
    prediction <-   data.frame(rat1, rat2, rat3, rat4, rat5)
    #print(prediction)
    rows <- nrow(prediction)
    num <- 1:rows
    for(k in num)
    {
      sum <- prediction[k,1] + prediction[k,2]+ prediction[k,3]+ prediction[k,4]+ prediction[k,5]
      scale <- 1 / sum
      prediction[k,1] <-  prediction[k,1] * scale
      prediction[k,2] <-  prediction[k,2] * scale
      prediction[k,3] <-  prediction[k,3] * scale
      prediction[k,4] <-  prediction[k,4] * scale
      prediction[k,5] <-  prediction[k,5] * scale
    }
    print(prediction)
  }
  if(probsFitOut$method == 3)
  {
    print("we are predicting KNN")
    maxRating <- probsFitOut$data_1
    data <- probsFitOut$data_2
    specialArgs <- probsFitOut$data_3
    
    pred <- matrix(nrow = dim(newXs)[1], ncol= maxRating)
    for(i in 1:(maxRating-1)){
      for(j in 1: dim(newXs)[1]){
        for(k in 1: length(data[[i]])){
          if(data[[i]][[k]]$userID == newXs$userID[j]){
            pred[j,i] <- predict(data[[i]], list(data[[i]][[k]]), newXs$itemID[j],specialArgs)
          }
        }
      }
    }
    pred[, maxRating] <- 1-rowSums(pred[,1:maxRating-1])
    pred <- as.matrix(pred)
    print(pred)
  }
  if(probsFitOut$method == 4)
  {
    output <- predict(probsFitOut$data, newXs, type = "prob")
    print(output)
  }
  
}



