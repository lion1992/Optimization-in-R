### gradient of the negative loglikelihood function ###

grad<-function(X,y,b){
 
  ## get the sigmoid function ##
  sigmoid<-function(z){
    sig<-1/(1+exp(-z));
    return (sig);
  } 
  
  s=t(X)%*%(y-sigmoid(X%*%b)); #get the gradient of the loglikelihood function.
  delta=-s; # get the gradient of the "loss" function, which is the negative of likelihood function.
  return (delta);
}