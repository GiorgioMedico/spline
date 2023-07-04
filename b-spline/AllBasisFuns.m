function [B] = AllBasisFuns(u,p,U)
n = length(U)-p-1;
B = zeros(1,n);
i = WhichSpan3(u,U,p);

if (i >=length(U)-p)
    B(n)=1;
else
    B1(1)=1;
    for j=1:p
        DL(j) = u - U(i+1-j);
        DR(j) = U(i+j) - u;
        acc = 0.0;
        for r=1:j
            temp = B1(r)/(DR(r) + DL(j-r+1));
            B1(r) = acc + DR(r)*temp;
            acc =DL(j-r+1)*temp;
        end
        B1(j+1) = acc;
    end
    for j=1:1:(p+1)
        B(i-p-1+j) = B1(j);
    end
end