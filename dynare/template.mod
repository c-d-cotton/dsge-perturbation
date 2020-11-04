//Generated from python2dynare function.
//Written for Dynare 4.
 
var logc logk loga;
// predetermined_variables ;
varexo e;

parameters beta rho alpha delta sigma;

// parameters_define;
beta=0.95;
rho=0.9;
alpha=0.3;
delta=0.1;
sigma=0.1;
ass = 1;
kss =((ass*alpha)/(1/(beta)+delta-1))^(1/(1-alpha));
css = ass*kss^alpha-delta*kss;
logass = log(ass);
logkss = log(kss);
logcss = log(css);
// parameters_define_end;

model;
exp(logc)^(-1)=beta*exp(logc(+1))^(-1)*(exp(loga(+1))*alpha*exp(logk)^(alpha-1)+1-delta);
exp(logc)+exp(logk)=exp(loga)*exp(logk(-1))^alpha+(1-delta)*exp(logk(-1));
loga = rho*loga(-1)+e;
end;

initval;
logk = logkss;
logc = logcss;
loga = logass;
end;

steady;

// shocks;
// var e; stderr 1;
// end;

// simulation_command;
stoch_simul(order=1);
// simulation_command_end;
