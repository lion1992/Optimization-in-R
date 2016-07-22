### negative loglikelihood function ###


cost <- function(X,y,b){
  n=length(y);
  
  sigmoid<-function(X,b){
    sig<-1/(1+exp(-X%*%b));
    return (sig);
  } 
  
  l=t(y)%*%X%*%b+sum(log(1-sigmoid(X,b)));
  #l=t(y)%*%log(1/(1+exp(-X%*%b)))+(1-t(y))%*%log(1-1/(1+exp(-X%*%b)));
  cost=-l;
  return (cost);
}