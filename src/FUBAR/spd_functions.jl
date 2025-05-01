expmap_spd(X,V) = begin
    Xh = sqrt(X);  invXh = inv(Xh)
    Xh * exp(invXh*V*invXh) * Xh
end

logmap_spd(X,Y) = begin
    Xh = sqrt(X);  invXh = inv(Xh)
    Xh * log(invXh*Y*invXh) * Xh
end

geo_drift(X) = -(size(X,1)+1)/2 * X
