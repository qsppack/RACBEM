%--------------------------------------------------------------------------
% Generate phase factors used in RACBEM, the example is about solving QLSP
%
% Please run Remez.ipynb to get a polynomial approximation beforehand. Make
% sure the data file storing coefficients are propeerly saved locally and
% that used in load() function is correct.
%
% You need to install qsppack before running this code. Several functions
% provided in the package are used.
%
% parameters
%     kappa: condition number
%     deg: degree of the polynomial approximation
%     criteria: stop criteria of qsppack, default 1e-12
%
%--------------------------------------------------------------------------
% References:
%     Yulong Dong, Lin Lin. Random circuit block-encoded matrix and a proposal 
%         of quantum LINPACK benchmark.
% Authors:
%     Yulong Dong     dongyl (at) berkeley (dot) edu
%     Lin Lin         linlin (at) math (dot) berkeley (dot) edu
% Version: 1.0
% Last revision: 06/2020
%--------------------------------------------------------------------------

function GeneratePhi(kappa,deg)

% setup parameters in qsppack
criteria = 1e-12;
% even oveerall parity because h(x) is even
parity = 0;
opts.criteria = criteria;

% load data pre-calculated by Remez.ipynb
load("coef_"+int2str(kappa)+"_"+int2str(deg)+".mat","coef");

% compute polynomial approximation error of Remez algorithm
x = linspace(0,1,1000)';
f = @(x) ChebyCoef2Func(x, coef, parity, true);
fx = f(x);
fr = @(x) 1./(kappa*((1-1/kappa)*x.^2+1/kappa));
frx = fr(x);

plot(x, fx, x, frx);
pause

% slightly rescale the polynimial
fac = 1.2 * max(fx);
coef = coef / fac;
% total scaling factor compared with 1/x \circ h
% an extra kappa is referred to the definitiion of inversex in Remez.ipynb
total_fac = fac * kappa;

% print basic information of polynomial approximation
fprintf('approx error (inf) of coef = %g\n', norm(fx-frx,'inf'));
fprintf('extra scaling factor = %g\n', fac);
fprintf('total scaling factor = %g\n', total_fac);

% find phase factors via qsppack
[phi,out] = QSP_solver(coef,parity,opts);
parity_label = ["even" "odd"];
fprintf("- Info: \t\tQSP phase factors --- solved by L-BFGS\n")
fprintf("- Parity: \t\t%s\n- Degree: \t\t%d\n", parity_label(parity+1), deg);
fprintf("- Iteration times: \t%d\n", out.iter);
fprintf("- CPU time: \t%.1f s\n", out.time);

% post processing phase factors to be used, SU(2) phi -> QSP varphi
% the correction is (pi/4,pi/2,...,pi/2,pi/4)
phi_proc = phi + pi/2;
phi_proc(1) = phi_proc(1) - pi/4;
phi_proc(end) = phi_proc(end) - pi/4;

% compute QSP approximation error wrt polynomial and 1/x \circ h
% QSP
yqsp = zeros(size(x));
% polynomial
ftotx = zeros(size(x));

% coef is scaled
ftot = @(x) ChebyCoef2Func(x,coef,parity,true);
% 1/x \circ h
frx = fr(x)/fac;

for jj = 1:length(x)
    yqsp(jj) = QSPGetUnitary(phi,x(jj));
    ftotx(jj) = ftot(x(jj));
end

% plot the error at different x
semilogy(x,abs(ftotx-frx))
pause

fprintf('approx error (inf) of polynomial = %.3e\n', norm(ftotx-yqsp,inf));
fprintf('approx error (inf) of g circ h = %.3e\n', norm(frx-yqsp,inf));

% save data
data = [phi_proc; total_fac; norm(frx-yqsp,inf)];
dlmwrite("phi_inv_"+num2str(kappa)+".txt", data,'precision','%25.15f')

close all

end
