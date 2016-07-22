### IRLS ###
irls<-function(X,y,b,tol,MAXITER){
  
  ## get the sigmoid function. ##
  sigmoid<-function(z){
    sig<-1/(1+exp(-z));
    return (sig);
  } 
  iter=0; # initialize iter.
  thetaold<- b; # initialize thetaold.
  w=as.vector(exp(X%*%thetaold)/((1+exp(X%*%thetaold))^2))
  W=diag(w); # initialize the W matrix, which is a diagonal matrix with diagonal elements being exp(X*theta)/((1+exp(X*theta))^2)
  thetanew <- solve(t(X) %*% W%*%X) %*% t(X) %*% W %*% (X%*%thetaold+solve(W)%*%(y-sigmoid(X%*%thetaold)))
  # initialize thetanew = (X.T*W*X)^(-1) * (X.T*W*Z).
  
  ## the loop would stop if it satisfies one of the two conditions: 
  ###1. the euclidean norm of the change of the theta vector is smaller than a certain value;
  ###2. the number of iteration reaches a pre-set maximum.
  while(norm(thetaold-thetanew,type="f")>tol && iter<=MAXITER)
  {
    thetaold=thetanew;
    w=as.vector(exp(X%*%thetaold)/((1+exp(X%*%thetaold))^2))
    W=diag(w); # update W matrix.
    z=X%*%thetaold+solve(W)%*%(y-sigmoid(X%*%thetaold)) # update matrix z=X*thetaold+W^(-1)*(y-sigmoid(X*thetaold)).
    thetanew=solve(t(X)%*%W%*%X) %*% t(X)%*%W%*%z #update thetanew = (X.T*W*X)^(-1) * (X.T*W*Z)
    iter=iter+1; # update iter.
  }

  return (list(thetanew,iter));
} 