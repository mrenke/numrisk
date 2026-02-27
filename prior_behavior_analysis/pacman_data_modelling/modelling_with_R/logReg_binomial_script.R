## model numerical precision


pati = "/Users/mrenke/Desktop/dyscalc_study/modeling_PacmanData"
setwd(pati)
library(runjags)
library("haven")                                     
library("sjlabelled")
library(LaplacesDemon)

Data_Pacman <- read_sav("Data_Pacman.sav")
Data_Pacman <- Data_Pacman[Data_Pacman$Einschluss == 1,]
diff_levels = cbind("small_diff,", "medium_diff", "large_diff")
ratios = cbind(0.91, 0.83,0.7)
data_all = cbind(Data_Pacman$P_NMB_ACC_S, Data_Pacman$P_NMB_ACC_M, Data_Pacman$P_NMB_ACC_L)
group = Data_Pacman$Group # 1= DD, 0 = CC


#data = data_all[group == '1',]
data = data_all

Nsubj = nrow(data)
Ntotal = length(data)
y = rep(0,1,Ntotal)
subIdx = rep(0,1, Ntotal)

for (i in 1:Nsubj) {
  j = (i-1)*3 + 1
  y[j] = data[i,1]
  y[j+1] = data[i,2]
  y[j+2] = data[i,3]
  subIdx[j:(j+2)] = rep(i,3)
  
  }

x = rep(ratios,Nsubj)

###############

groupIdx = as.numeric(remove_all_labels(group))
groupIdx[groupIdx == 0] = 2 ## CC is now 2 instead of 0
Ngroups = length(unique(groupIdx))

n = round(round(mean(Data_Pacman$P_NMB_AnzahlTrials_Total))/ 3 )
n_trials = n
x_log = log(x)
y_bin = n - round(y*n)
dat <- dump.format(list(y=y_bin, x=x_log, subIdx=subIdx,groupIdx = groupIdx, Ntotal=Ntotal, Nsubj=Nsubj, Ngroups = Ngroups, n_trials = n_trials))


inits1 <- dump.format(list( xbeta.mu=0, .RNG.name="base::Super-Duper", .RNG.seed=99999 ))
inits2 <- dump.format(list( xbeta.mu=1, .RNG.name="base::Wichmann-Hill", .RNG.seed=1234 ))
inits3 <- dump.format(list( xbeta.mu=2, .RNG.name="base::Mersenne-Twister", .RNG.seed=6666 ))

# Tell JAGS which latent variables to monitor
monitor = c("xbeta.mu", "xbeta.n.mu")

# Run the function that fits the models using JAGS
results <- run.jags(model="model_logReg_binomial.txt",
                    monitor=monitor, data=dat, 
                    n.chains=3, inits=c(inits1, inits2, inits3), plots = FALSE, burnin=5000, sample=1000, thin=5)

chains1 = rbind(results$mcmc[[1]], results$mcmc[[2]], results$mcmc[[3]])
plot(density(chains1[,"xbeta.n.mu[1]"]), col = "blue",  xlim=c(0,20), ylim=c(0,0.6))
par(new=TRUE)
plot(density(chains1[,"xbeta.n.mu[2]"]), col = "red",  xlim=c(0,20), ylim=c(0,0.6))
legend(-4, 70, legend = c("DD","control"), col = c("blue","red" ), lty=1, cex=0.8)


post_DD = chains1[,"xbeta.n.mu[1]"]
post_CC = chains1[,"xbeta.n.mu[2]"]
d = post_DD-post_CC
plot(density(d)) 

source('/Users/mrenke/Desktop/bayesianDataAnalysis&Behavior/Assignment1_for_OLAT/HDIofMCMC.r')
hdi = HDIofMCMC(d , credMass=0.95)
abline(v=median(d), col="red", lwd=2)
abline(v=hdi[1], lty=2, col="red", lwd=2)
abline(v=hdi[2], lty=2, col="red", lwd=2)  


###
## write tables
n = round(round(mean(Data_Pacman$P_NMB_AnzahlTrials_Total))/ 3 )

data_f = cbind(subIdx, group, log(x), n - round(y*n), rep(n, length(x)))
colnames(data_f) <- c("subIdx", "group", 'log_ratio','n_choice_2', 'n_total')  
write.table(data_f, file="data.csv")
