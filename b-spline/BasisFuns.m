function [B] = BasisFuns(i,u,p,U)
if (i >=length(U)-p)
    B = zeros(1,p+1)
    B(p+1)=1;
else
    B(1)=1;
    for j=1:p
        DL(j) = u - U(i+1-j);
        DR(j) = U(i+j) - u;
        acc = 0.0;
        for r=1:j
            temp = B(r)/(DR(r) + DL(j-r+1));
            B(r) = acc + DR(r)*temp;
            acc =DL(j-r+1)*temp;
        end
        B(j+1) = acc;
    end
end