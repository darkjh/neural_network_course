function digits = name2digits(name)
% takes a string NAME and converts it into a pseudo-random selection of 4
% digits from 0-9.
%
% Example:
% name2digits('Felipe Gerhard')
% returns: [0 4 5 7]

name = lower(name);

if length(name)>25
    name = name(1:25);
end

primenumbers = primes(100); % first 25 primes

n = length(name);
s = 0;
for i=1:n
    s = s + primenumbers(i)*(double(name(i)))*2^i;
end

load('hash'); % has a variable "x" with all permutations of the digits
t = mod(s,size(x,1));

digits = sort(x(t,:));

end % function