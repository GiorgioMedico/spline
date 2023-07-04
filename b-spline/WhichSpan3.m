function [i] = WhichSpan3(u,U,p)

if ~exist('p', 'var')
   p = 3; %degree of the spline 
end

high = length(U)-p; 
low = p+1;
if (u>=U(high))
    i=high;
else
    mid = floor((high+low)/2);
    while ((u<U(mid))||(u>=U(mid+1)))
        if (u==U(mid+1))
            mid = mid+1;   % knot with multiplicity >1
        else 
            if (u > U(mid))
                low = mid;
            else
                high=mid;
            end
            mid = floor((high+low)/2);
        end
    end
    i=mid;
end

    
    