BFGS <- function(X, y, H0, tol, MAXITER){
  # Use BFGS to learn a logistic regression model for the given dataset.
  
  # Input:
  # X: the matrix of entries for the variables that the user wants to get the weights for;
  # y: matrix of entries of the labels (0/1);
  # H0: the value that the user wants the initial inverse hessian matrix to take (the same for all diagonal elements);
  # tol: convergence threshold used to detect convergence;
  # MAXITER: maximum number of iterations to perform;
  
  # Output:
  # Weights: the learned weights for the variables;
  # Number_of_Iterations: the number of iterations the learning process takes;
  # Final_Logloss: the final value of the logloss function;
  # Total_ElapsedTime: the total elapsed time(in seconds) the learning process takes. 
  
  iter=0; 
  
  # get the size of the X matrix;
  n = dim(X)[1];
  d = dim(X)[2];
  # initialize a vector for the weights;
  thetaold=matrix(0, nrow=d, ncol = 1);  
  # initialize a diagonal matrix w/ diagonal elements being of the input values;
  H = H0*diag(1, nrow = d, ncol = d);
  done = FALSE;
  I = diag(1, nrow = d, ncol = d);
  err=c(); 
  loss= c();
  time = c();
  # set the clock;
  ptm <- proc.time();
  while(done == FALSE){
    # Compute the P(y=1|X,weights) with weights from the previous iteration;
    pold = 1/(1+exp(-X %*% thetaold));
    # Compute the gradient of the logloss function w/ weights learned from the previous iteration;
    g0 = grad(X,y,thetaold);
    # Compute the absolute direction of the step;
    direction = H %*% g0;
    # get step size using backtracking line search;
    t = backtracking(X, y, thetaold, g0, direction, 0.01, 0.5)
    # update weights;
    thetanew=thetaold- t * direction; 
    # Compute the P(y=1|X,weights) with weights just learned.
    pnew = 1/(1+exp(-X %*% thetanew));
    iter=iter+1; # update iter.
    
    # check if the convergence criterion are satisfied;
    # if yes, exit the loop and make the weights just learned the final weights;
    if(mean(abs(pnew - pold))<tol || iter > MAXITER )
    {
       done = TRUE;
       weights = thetanew;
    }
    # if not, update the approximating inverse hessian matrix with following procedures;
    else
    {
      # compute the gradient of the logloss function with the weights just learned;
      g1 = grad(X,y,thetanew);
      # compute the difference between the new gradient and the gradient computed in the beginning of the loop;
      gd = g1 - g0;
      # compute the difference between the weights just learned and the weights from the previous iteration;
      s = thetanew - thetaold;
      # update the approximating inverse hessian matrix;
      p = as.numeric(1/(t(gd) %*% s));
      H = (I - p * s %*% t(gd)) %*% H %*% (I - p * gd%*%t(s)) +  p * s %*% t(s);
      thetaold = thetanew;
    }
    # compute the accuracy and the value of logloss function with weights just learned;
    chat=pnew>0.5;
    acc=100*sum(y==chat)/length(y);
    err=c(err,acc);
    logloss=sum(log(1+exp(X%*%thetanew)))-t(y)%*%X%*%thetanew;
    loss=c(loss, logloss);  
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
  return (list(Weights = weights, Prob = pnew, Number_of_Iterations = iter, Final_LogLoss = logloss, Total_ElapsedTime = elapsed, FinalAccuracy = acc));
}
