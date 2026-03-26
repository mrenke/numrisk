

dat = read.table("NMB_merged.txt", sep = ",", header = TRUE)

diff_levels = cbind("small_diff,", "medium_diff", "large_diff")
ratios = cbind(0.91, 0.83,0.7)

dat = dat[dat$Condition.paramod != 0,] #filter out control condition
#filter subjects
remSubs = as.numeric(Data_Pacman$VP[Data_Pacman$Einschluss == 0])
for (sub in 1:length(remSubs)){
  dat = dat[dat$Sub != remSubs[sub],]
  
}
Ntotal = nrow(dat)

choice = dat$Images.ACC#y
Nsubj = length(unique(dat$Subject))

#groupIdx = rep(NaN,Ntotal)
subIdx = rep(NaN,Ntotal)
x = rep(NaN, Ntotal)

Subs = unique(dat$Subject)
sub_counter = 1
for (i in 1:length(x)) {
  x[i] = ratios[dat[i,'Condition.paramod']]
  
  # add 1 to sub_counter when new line is with different subject
  if (dat$Subject[i] != Subs[sub_counter]) { 
    sub_counter = sub_counter + 1
  }
  subIdx[i] = sub_counter
  #groupIdx[i] = as.numeric(group_VP_list[group_VP_list[,1] == Subs[sub_counter],2]) # some subs are not in Data_Pacman..
  
}

Data_Pacman <- read_sav("Data_Pacman.sav")
Data_Pacman <- Data_Pacman[Data_Pacman$Einschluss == 1,]

group_VP_list = cbind(as.numeric(Data_Pacman$VP), Data_Pacman$Group)  # 1= DD, 0 = CC
groupIdx = rep(NaN,Nsubj)
for (sub_counter in 1:Nsubj){
  groupIdx[sub_counter] = as.numeric(group_VP_list[group_VP_list[,1] == Subs[sub_counter],2])  
}
groupIdx[groupIdx == 0] = 2 ## CC is now 2 instead of 0

Ngroups = length(unique(groupIdx))
log_ratio = -log(x) #x

  
jags_data <- dump.format(list(y=choice, x=log_ratio, subIdx=subIdx,groupIdx = groupIdx, Ntotal=Ntotal, Nsubj=Nsubj, Ngroups = Ngroups))

inits1 <- dump.format(list( .RNG.name="base::Super-Duper", .RNG.seed=99999 ))
inits2 <- dump.format(list( .RNG.name="base::Wichmann-Hill", .RNG.seed=1234 ))
inits3 <- dump.format(list( .RNG.name="base::Mersenne-Twister", .RNG.seed=6666 ))

monitor = c("xbeta.s", "xbeta.n.mu", "pr")

# Run the function that fits the models using JAGS
results <- run.jags(model="model_logReg_bernoulli.txt",
                    monitor=monitor, data=jags_data, 
                    n.chains=3, inits=c(inits1, inits2, inits3), plots = FALSE, burnin=5000, sample=1000, thin=5)

chains1 = rbind(results$mcmc[[1]], results$mcmc[[2]], results$mcmc[[3]])

plot(density(chains1[,"xbeta.n.mu[1]"]), col = "blue", xlim=c(0,20), ylim=c(0,0.6))
par(new=TRUE)
plot(density(chains1[,"xbeta.n.mu[2]"]), col = "red",  xlim=c(0,20), ylim=c(0,0.6))
legend(0,0.6, legend = c("DD","control"), col = c("blue","red" ), lty=1, cex=0.8)

# from Piazza 2010: "2w represents the percentage difference between two numerosities that is necessary to perceive them as different with 95% confidence"
Ws_DD = logit(0.95)/(chains1[,"xbeta.n.mu[1]"] * 2)
Ws_CC = logit(0.95)/(chains1[,"xbeta.n.mu[2]"] * 2)
xlim = c(0,1)
ylim=c(0,30)
plot(density(Ws_DD), col = "blue", xlim=xlim, ylim=ylim)
par(new=TRUE)
plot(density(Ws_CC), col = "red",  xlim=xlim, ylim=ylim)
legend(0,0.6, legend = c("DD","control"), col = c("blue","red" ), lty=1, cex=0.8)

# plot difference
post_DD = chains1[,"xbeta.n.mu[1]"]
post_CC = chains1[,"xbeta.n.mu[2]"]
d = post_DD-post_CC
plot(density(d), main = 'Difference in slope Dyslexic minus Controls') 

source('/Users/mrenke/Desktop/bayesianDataAnalysis&Behavior/Assignment1_for_OLAT/HDIofMCMC.r')
hdi = HDIofMCMC(d , credMass=0.95)
abline(v=median(d), col="red", lwd=2)
abline(v=hdi[1], lty=2, col="red", lwd=2)
abline(v=hdi[2], lty=2, col="red", lwd=2)  



## test mean accuracy between two data formats

x[choice == 0] #false answers, 312
mean(x[choice == 0]) # 0.833 ?!

N_subs = length(unique(dat$Subject))
meanACCs = matrix(0,N_subs, 3)
               
for (sub in 1:length(unique(dat$Subject))) {
  sub_fil = dat[dat$Subject == unique(dat$Subject)[sub], ]
  meanACCs[sub, 1] = mean(sub_fil[sub_fil$Condition.paramod ==1,'Images.ACC'])
  meanACCs[sub, 2] = mean(sub_fil[sub_fil$Condition.paramod ==2,'Images.ACC'])
  meanACCs[sub, 3] = mean(sub_fil[sub_fil$Condition.paramod ==3,'Images.ACC'])
}

nmb_df = cbind(unique(dat$Subject), groupIdx, meanACCs)
Data_Pacman <- Data_Pacman[Data_Pacman$Einschluss == 1,]
pacman_df = cbind(as.numeric(Data_Pacman$VP), Data_Pacman$Group, Data_Pacman$P_NMB_ACC_S, Data_Pacman$P_NMB_ACC_M, Data_Pacman$P_NMB_ACC_L)

##

xs = seq(-1,1, 0.1)
plot(xs, invlogit(xs))


