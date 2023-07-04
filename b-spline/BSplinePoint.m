function s = BSplinePoint(u,U,p,P)

%i = WhichSpan(u,U,p);
i = WhichSpan(u,U,3);
B = BasisFuns(i,u,p,U);
sizeP = size(P);
ComponentsP = sizeP(1);
s = zeros(ComponentsP,1);
for j = 1:(p+1)
    s = s + P(:,i-p+j-1)*B(j);
end
    
    