close all
clear variables
clc

g = 9.8;
mc = 1;
mp = 0.1;
l = 0.5;

A = [0 1 0 0; 0 0 -mp*(mp * (g-l) + mc*g)/((mc+mp)*((4/3) * mc + (1/3) * mp)) 0; 0 0 0 1; 0 0 (mp*(g-l) + mc * g)/(l*((4/3) * mc + (1/3) * mp)) 0]
B = [0 (1/(mc + mp) - mp/((mc + mp) * ((4/3) * mc + (1/3) * mp))) 0 (-1/(l * ((4/3) * mc + (1/3) * mp)))]
C = [1 0 0 0; 0 0 1 0]
At = transpose(A);
Bt = transpose(C);
Ct = transpose(B);