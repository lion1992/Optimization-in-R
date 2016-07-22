LBFGS <- function (X, y, M, H0, tol, MAXITERATIONS){
  # Use Limited-Memory-BFGS to learn a logistic regression model for the given dataset.
  
  # Input:
  # X: the matrix of entries for the variables that the user wants to get the weights for;
  # y: matrix of entries of the labels (0/1);
  # M: the maximum number of pairs of the d-vectors to be stored; 
  # H0: the value that the user wants the initial inverse hessian matrix to take (the same for all diagonal elements);
  # tol: convergence threshold used to detect convergence;
  # MAXITER: maximum number of iterations to perform;
  
  # Output:
  # Weights: the learned weights for the variables;
  # Number_of_Iterations: the number of iterations the learning process takes;
  # Final_Logloss: the final value of the logloss function;
  # Total_ElapsedTime: the total elapsed time(in seconds) the learning process takes. 
  
  iter=0; # initialize iter.
  # get the size of the X matrix;
  n = dim(X)[1];
  d = dim(X)[2];
  # initialize a vector for the weights;
  thetaold=matrix(0, nrow=d, ncol = 1);  
  # initialize a diagonal matrix w/ diagonal elements being of the input values;
  H0 = H0*diag(1, nrow = d, ncol = d);
  done = FALSE;
  I = diag(1, nrow = d, ncol = d);
  err=c(); 
  loss= c();
  time = c();
  
  # initialize the matrix storing the d-vectors {Sk}, in which each column is the difference between the weights just learned and the weights from previous iteration;
  Sk <- matrix( , nrow = d, ncol = 1); 
  # initialize the matrix storing the d-vectors {yk}, in which each column is the difference between the gradients with weights just learned and the weights from previous iteration;
  yk <- matrix( , nrow = d, ncol = 1); 
  # set the clock;
  ptm <- proc.time();
  
  # compute the gradient with the initial weights (or weights from iter 0);
  g0 = grad(X, y, thetaold);
  # compute the absolute direction of the step;
  direction = H0 %*% g0;
  # compute the step size using backtracking line search;
  t = backtracking(X, y, thetaold, g0, direction, 0.01, 0.8)
  # update and get the weights of iter 1;
  thetanew = thetaold - t*direction;
  # compute S1 (Weights1 - Weights0) and store it in the first column of {Sk};
  Sk[,1] = thetanew - thetaold;
  # compute the gradient with the new weights just learned;
  g1 = grad(X, y, thetanew);
  # compute y1 (gradient1 - gradient0);
  yk[,1] = g1 - g0;
  # compute p1 = 1/(y1' * S1);
  p = as.numeric(1/(t(yk[,1]) %*% Sk[,1]));
  thetaold = thetanew;
  iter = iter + 1;
  # compute the value of logloss function with Weights1;
  logloss=sum(log(1+exp(X%*%thetanew)))-t(y)%*%X%*%thetanew;
  loss=c(loss, logloss);  
  # get and store the time elapsed so far into a vector;
  elapsed = (proc.time()-ptm)[3];
  time = cbind(time, elapsed);
  
  while (done == FALSE){
    # Compute the P(y=1|X,weights) with weights from the previous iteration (iter i-1);
    Pold = 1/(1+exp(-X %*% thetaold));
    # Compute the gradient (i) of the logloss function w/ weights learned from the previous iteration(iter i-1);
    g0 = grad(X, y, thetaold);
    # get the number of columns in the {Sk} matrix;
    m = dim(Sk)[2];
    # initialize q = gradient (i);
    q = g0;
    a = matrix( , nrow=1, ncol=m);
    
    # Compute H(i) * g(i) by rolling out the update equation used in BFGS;
    # Compute the right product;
    for(i in 1:m){
      a[i] = p[m-i+1] * t(Sk[, m-i+1])%*% q;
      q = q - a[i] * yk[, m-i+1];
    }
    ## adjust the H0;
    H0 = as.numeric((t(yk[,m]) %*% Sk[,m])/(t(yk[,m])%*% yk[,m])) * I;
    r = H0 %*% q;
    # compute the left product;
    for (j in 1:m){
      beta = p[j] * t(yk[,j]) %*% r;
      r = r + Sk[,j] * (a[m-j+1] - beta);
    }
    
    # get the absolute step directon 
    direction = r;
    # use backtracking line search to compute the appropriate step size
    t = backtracking(X, y, thetaold, g0, direction, 0.01, 0.8)
    # update the weight;
    thetanew = thetaold - t*direction;
    iter = iter + 1;
    # Compute and store the difference of weights S(i+1) into the matrix {Sk}
    Skjunk = (thetanew - thetaold);
    Sk = cbind(Sk, Skjunk);
    # Compute and store the difference of gradients y(i+1) into the matrix {yk}
    g1 = grad(X, y, thetanew);
    ykjunk = g1 - g0;
    yk = cbind(yk, ykjunk);
    # compute and store p(i+1) into the vector p;
    pjunk = as.numeric(1/(t(ykjunk)) %*% Skjunk);
    p = cbind (p, pjunk);
    thetaold = thetanew;
    # Compute the P(y=1|X,weights) with weights just learned;
    Pnew = 1/(1+exp(-X %*% thetanew));
    # check for convergence
    ## if yes, stop the loop;
    if(mean(abs(Pnew - Pold))<tol || iter > MAXITERATIONS )
    {
      done = TRUE;
      weights = thetanew;
    }
    # if not, check if the number of columns in {yk} exceeds M, that is, to check if the number of 
    ##pairs of the d-vectors has exceeds the preset M.
    else
    { 
      # if not, delete the first column / element in {Sk}, {yk} and p;
      if(dim(yk)[2] > M){
        Sk = Sk[, -1];
        yk = yk[, -1];
        p = p[1:M+1];
      }
    }
    # compute the accuracy and the value of logloss function with weights just learned;
    chat=Pnew>0.5;
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
  plot(time, loss, type = "l", xlab = "Time Elapsed", ylab = "Logloss", main = "Logloss vs. Time Elapsed");
  # plot the Accuracy vs. Iterations;
  plot(2:iter, err, type = "l",xlab = "Number of Iterations", ylab = "Accuracy", main = "Accuracy vs. Number of Iterations");
  
  return (list(Weights = weights, Number_of_Iterations = iter, Final_LogLoss = logloss, Total_ElapsedTime = elapsed));
  
  
}