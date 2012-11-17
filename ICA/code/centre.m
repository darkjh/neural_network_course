function res = centre(x)
    [n,~] = size(x);
   for i = 1:n
       x(i,:) = x(i,:)-mean(x(i,:));
   end 
   res = x;
end