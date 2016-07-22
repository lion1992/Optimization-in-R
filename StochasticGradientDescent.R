SGD <- function(X, y, t, M, epsilon, maxiterations){
  # Use Stochastic Gradient Descent to learn a logistic regression model for the given dataset.
  
  # Input:
  # X: the matrix of entries for the variables that the user wants to get the weights for;
  # y: matrix of entries of the labels (0/1);
  # t: the step size.
  # M: the size of the mini-batch;
  # tol: convergence threshold used to detect convergence;
  # MAXITER: maximum number of iterations to perform;
  
  # Output:
  # Weights: the learned weights for the variables;
  # Number_of_Iterations: the number of iterations the learning process takes;
  # Final_Logloss: the final value of the logloss function;
  # Total_ElapsedTime: the total elapsed time(in seconds) the learning process takes. 
  
  # get the size of the X matrix;
  n = dim(X)[1];
  d = dim(X)[2];
  # initialize a vector for the weights;
  b=matrix(0, nrow=d, ncol = 1);  
  thetaold = b; 
  iter=0; 
  done=FALSE;
  err=c(); 
  loss= c();
  time = c();
  # set the clock
  ptm <- proc.time();
  while (done == FALSE){
    # Compute the P(y=1|X,weights) with weights from the previous iteration;
    Pold = 1/(1+exp(-X %*% thetaold));
    # randomly sample without replacement a vector of size M from 1 to n and set it to be the index;
    index = sample(1:n, M);
    # use the index to generate a subset of X and y of size M;
    Xsub = X[index,];
    ysub = y[index];
    # get the subset of Pold corresponding to the index;
    poldsub = Pold[index];
    # update the gradient and the weight;
    gradient = grad(Xsub, ysub, thetaold);
    thetanew = thetaold - t * gradient;
    # Compute the P(y=1|X,weights) with weights just learned.
    Pnew = 1/(1+exp(-X %*% thetanew));
    # Get the average change of Pi's;
    delta = mean(abs(Pnew-Pold));
    iter = iter + 1;
    # compute the accuracy and the value of logloss function with weights just learned;
    chat=Pnew>0.5;
    acc=100*sum(y==chat)/length(y);
    err=c(err,acc);
    logloss=sum(log(1+exp(X%*%thetanew)))-t(y)%*%X%*%thetanew;
    loss=c(loss, logloss);
    #check for convergence
    if(delta<epsilon || iter>maxiterations){
      done=TRUE;
      weights=thetanew;
    }
    else{  
      thetaold=thetanew;
    }
    # get and store the time elapsed so far into a vector;
    elapsed = (proc.time()-ptm)[3];
    time = cbind(time, elapsed);
  }
  
  par(mfrow=c(3,1));
  # plot the Logloss vs. Iterations;
  plot(1:iter, loss, type = "l", xlab = "Number of Iterations", ylab = "Logloss", main = "Logloss vs. Number of Iterations");
  # plot the Logloss vs. Time Elapsed;
  plot(time, loss, type = "l",  xlab = "Time Elapsed", ylab = "Logloss", main = "Logloss vs. Time Elapsed");
  # plot the Accuracy vs. Iterations;
  plot(1:iter, err, type = "l",xlab = "Number of Iterations", ylab = "Accuracy", main = "Accuracy vs. Number of Iterations");
  return (list(Weights = weights, Number_of_Iterations = iter, Final_LogLoss = logloss, Total_ElapsedTime = elapsed));
}