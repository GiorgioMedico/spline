 clear all
 close all
 clc
 dt = 0.01;
 LW=1.5;
 Ts = 0.01; % sampling time
 
% desired via-points
Q = [0 1 2 4 5 6 7 8 9 10;
     0 4 1 3 2 3 0 -1 -1 2;
     0 0 0 0 0 0 0 0 0 0];


figure(1)
hold on
plot3(Q(1,:),Q(2,:),Q(3,:),'o','MarkerFaceColor','b','MarkerSize',5)
plot3(Q(1,:),Q(2,:),Q(3,:),'ob','MarkerSize',5)
grid on
xlabel('x')
ylabel('y')
zlabel('z')
%view(-75,18)
view(2)
N=5; % Number of control points   4 <= N 
m=size(Q,2);
W = eye(m-2);
l=0.000000; %smoothing parameter: l=0 --> approximating spline
[spline,P] = ApproximatingSmoothingBSpline(Q,W,N,l,Ts);
plot(spline(1,:),spline(2,:),'r')
plot(P(1,:),P(2,:),'dg')