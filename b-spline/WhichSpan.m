function [i] = WhichSpan(u,U,p)

if ~exist('p', 'var')
   p = 3; %degree of the spline 
end

high = length(U)-p; 
low = p+1;
if (u >=U(high))
    mid = high;  
else
    mid = floor((high+low)/2);
    while ((u<U(mid))||(u>=U(mid+1)))
        if (u==U(mid+1))
            mid = mid+1;   % kont with multiplicity >1
        else 
            if (u > U(mid))
                low = mid;
            else
                high=mid;
            end
            mid = floor((high+low)/2);
        end
    end
end
i=mid;
if (i>length(U)-p-1)|(u>=U(end))
    i=length(U)-p-1;
end
    
    