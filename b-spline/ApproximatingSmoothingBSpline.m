function [s,Pout] = ApproximatingSmoothingBSpline(Q,W,N,l,dt)
% [s,Pout] = ApproximatingSmoothingBSpline(Q,W,N,l,dt)

sizeQ = size(Q);
m=sizeQ(2); % number of via points
%n=floor(m/N)+1; %number of control points (first and last equal to the first and last via-point)
n=N;
%hatu = linspace(0,1,m); % Uniform knots distribution
hatu = NonUniformKnots(Q);
dim=sizeQ(1);
p=3;
d = m/(n-p);
u=zeros(1,n+p+1);

for j=1:1:n-p-1
    i=floor(j*d);
    a = j*d-i;
    u(p+j+1)=(1-a)*hatu(i)+a*hatu(i+1);
end

for k=1:1:p+1
u(n+k) = 1;
end

U = u
N=[];
R_m=[];
for k=2:1:(m-1)
    B = AllBasisFuns(hatu(k),3,u);
    R(:,k-1) = Q(:,k) - B(1) * Q(:,1) - B(n)*Q(:,m); % interpolation of the first and lsta point
    N = [N ;B(2:(n-1))];
end

%% Approximating B-spline
for j =1:1:3
    P1(j,:)=(pinv(N)*R(j,:)')';
end

Pa = [Q(:,1) P1 Q(:,m)];

%% Acceleration definition for smoothing
P2R=zeros(n,n);
for k=1:1:n-2
    P2R(k,k) = 6/(U(k+4)-U(k+2))/(U(k+4)-U(k+1));
    P2R(k,k+1) = -6/(U(k+4)-U(k+2))/(U(k+4)-U(k+1))-6/(U(k+4)-U(k+2))/(U(k+5)-U(k+2));
    P2R(k,k+2) =  6/(U(k+4)-U(k+2))/(U(k+5)-U(k+2));   
end
C = P2R(1:n-2,2:n-1);
HP = zeros(3,n-2);
HP(:,1) = P2R(1,1)*Q(:,1);
HP(:,n-2) = P2R(n-2,n)* Q(:,m);

A=zeros(n,n);
for k=1:1:n
     if (k==1)
         A(1,1)= 2*(U(5)-U(4));
         A(1,2)= (U(5)-U(4));
     elseif (k==(n))
          A(k,k-1)= (U(k+3)-U(k+2));
          A(k,k)= 2*((U(k+3)-U(k+2)));         
     else
          A(k,k-1)= (U(k+3)-U(k+2));
          A(k,k)= 2*(U(k+4)-U(k+2));
          %A(k,k)= 2*((U(k+3)-U(k+2))+(U(k+4)-U(k+3)));
          A(k,k+1)= (U(k+4)-U(k+3));
     end
end
A1 = A(1:n-2,1:n-2);

%% Smoothing Approximating B-spline
Pint=inv(N'*W*N +l*C'*A1'*C)*(N'*W*R'-l*C'*A1'*HP');
Ps = [Q(:,1) Pint' Q(:,m)];
Pout = Ps;
%% spline computation
i=1;
t= 0;
s=[];
dim=3;
for t=0:dt:1
        s = [s, BSplinePoint(t,U,p,Pout(1:dim,:))];
        time(i) = t;
        i=i+1;
end