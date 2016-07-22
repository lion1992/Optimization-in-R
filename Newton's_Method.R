 Newton <- function(data,labels,epsilon,maxiterations){
   # Use Newton's method to learn a logistic regression model for the given dataset.

   # Input:
   # data: the matrix of entries for the variables that the user wants to get the weights for;
   # labels: matrix of entries of the labels (0/1);
   # epsilon: convergence threshold used to detect convergence;
   # maxiterations: maximum number of iterations to perform;
   
   # Output:
   # Weights: the learned weights for the variables;
   # Number_of_Iterations: the number of iterations the learning process takes;
   # Final_Logloss: the final value of the logloss function;
   # Total_ElapsedTime: the total elapsed time(in seconds) the learning process takes. 
   
    X=data;
    y=labels;
    
    # initialize a vector for the weights;
    b=matrix(0, nrow=length(X[1,]), ncol = 1);  
    thetaold = b; 

    iter=0; 
    done=FALSE;
    err=c(); 
    loss= c();
    n=length(y);
    t = c();
    
    # set the clock;
    ptm <- proc.time();
    while(done==FALSE){
      # Compute the P(y=1|X,weights) with weights from the previous iteration.
      Pold=1/(1+exp(-X %*% thetaold));
      # Compute the gradient of the log-loss objective function.
      grad=t(X) %*% (Pold-y);
      # Compute the diagonal matrix (with diagonal elements Pi(1-Pi))needed for the Hessian matrix. 
      w=as.vector(exp(X%*%thetaold)/((1+exp(X%*%thetaold))^2));
      W=diag(w); 
      # Update the Hessian matrix.
      hessian=t(X)%*% W %*% X;
      # update the weights;
      thetanew=thetaold-solve(hessian+10^(-6)*diag(length(X[1,])))%*%grad;
      
      # Compute the P(y=1|X,weights) with weights just learned.
      Pnew=1/(1+exp(-X%*%thetanew));
      
      # compute the average o fabsolute change of Pi's;
      deltas=abs(Pnew-Pold);
      delta=mean(deltas);
      iter=iter+1;
      
      # compute the accuracy and the value of logloss function with weights just learned;
      chat=Pnew>0.5;
      acc=100*sum(y==chat)/length(y);
      err=c(err,acc);
      logloss=sum(log(1+exp(X%*%thetanew)))-t(y)%*%X%*%thetanew;
      loss=c(loss, logloss);

      # check if the convergence criterion are satisfied;
      if(delta<epsilon || iter>maxiterations){
        done=TRUE;
        weights=thetanew;
      }
      else{  
      thetaold=thetanew;
      }
      # get and store the time elapsed so far into a vector;
      elapsed = (proc.time()-ptm)[3];
      t = cbind(t, elapsed);
    }
    
    par(mfrow=c(3,1));
    # plot the Logloss vs. Iterations;
    plot(1:iter, loss, type = "l", xlab = "Number of Iterations", ylab = "Logloss", main = "Logloss vs. Number of Iterations");
    # plot the Logloss vs. Time Elapsed;
    plot(t, loss, type = "l", xlab = "Time Elapsed", ylab = "Logloss", main = "Logloss vs. Time Elapsed");
    # plot the Accuracy vs. Iterations;
    plot(1:iter, err, type = "l",xlab = "Number of Iterations", ylab = "Accuracy", main = "Accuracy vs. Number of Iterations");
  return (list(Weights = weights, Number_of_Iterations = iter, Final_LogLoss = logloss, Total_ElapsedTime = elapsed))
    }
 
    