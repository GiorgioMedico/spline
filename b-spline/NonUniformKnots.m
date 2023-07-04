function x = NonUniformKnots(y)

d = sum(sqrt(sum((y(:,2:length(y))-y(:,1:length(y)-1)).^2)))

x(1) = 0;


for i=2:1:length(y)
   x(i)=x(i-1)+ sqrt(sum(((y(:,i) - y(:,i-1)).^2)))/d; 
end
