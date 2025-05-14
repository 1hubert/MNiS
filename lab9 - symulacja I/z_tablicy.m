clc, clear all, close all

tspan = [0 5];                        % wektor czasu
y0 = 0;                               % warunek poczatkowy
[t,y] = ode45(@wzorek, tspan, y0);    % calkujemy

plot(t, y, '-o')                      % wydrukuj wykres
axis([0 5 0 40])
grid on
xlabel('Czas')
ylabel('y')

function wy = wzorek(t,in)
	% dy/dt = 2t
    in = t;
    wy = 2*in;
end