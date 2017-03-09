import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

D = 200 # difficulty of share being a block

N_p = 5 # number of pools

N_m = 100 # number of miners

T1 = 1000 # time period for initialization
T2 = 100*T1 # time period for simulatio

trials = 25
rev_ratio_hop = trials*[0.]
rev_ratio_pplns = trials*[0.]
rev_ratio_fix = trials*[0.]

for ii in range(trials):

	# randomly assign computing power for each miner
	r = np.random.rand(N_m,1)
	alpha = r/sum(r)

	# assign a certain proportion of miners to be pool-hopping
	hop = 0.20 # so say first hop % of miners are hoppers, rest are not

	# revenue of each miner
	rev = np.zeros((N_m,1))

	# sample finder of each share
	share_winners = np.random.choice(N_m, T1+T2, p = alpha[:,0].tolist())

	# keep track of current population and share distribution of each pool
	curr_pools = [(N_m+1)*[0] for i in range(N_p)]
	# curr_pools[pool number] = [list with number of current shares owned by each miner and last number is current population] 

	# expected prices per share
	prices = N_p*[0]

	stable_comp_power = np.zeros((N_p,1))

	# current location of each miner
	loc = np.zeros((N_m,1))

	# initialize population of pools

	# assign each miner uniformly at random to each pool
	for k in range(N_m):
		p = np.random.randint(N_p)
		curr_pools[p][N_m] += 1
		loc[k] = p
		if k > hop*N_m:
			stable_comp_power[p] += alpha[k]

	found_blocks = np.zeros((N_p,1))

	# run T1 steps to get into an "initial" state
	# in these steps, no one can pool hop - populations fixed
	for i in range(T1):
		m_i = int(share_winners[i]) # miner of share
		p_i = int(loc[m_i]) # current pool of miner
		curr_pools[p_i][m_i] += 1 # give credit
		check_block = np.random.rand()
		if check_block <= 1./D:
			# share was a valid block
			found_blocks[p_i] += 1
			# update revenues
			tot = sum(curr_pools[p_i][:-1])
			for j in range(N_m):
				rev[j] += float(curr_pools[p_i][j])/tot

			# clear share history of pool
			curr_pools[p_i][:-1] = N_m*[0]

	# compute current expected price per share of each pool
	for i in range(N_p):
		prices[i] = 1./(D+sum(curr_pools[i][:-1]))

	tot_hops = 0
	loy_hops = 0

	# run T2 steps for simulation
	# people may pool hop after every found share
	for i in range(T2):
		m_i = int(share_winners[i]) # miner of share
		p_i = int(loc[m_i]) # current pool of miner
		curr_pools[p_i][m_i] += 1 # give credit
		check_block = np.random.rand()
		if check_block <= 1./D:
			# share was a valid block
			found_blocks[p_i] += 1
			# update revenues
			tot = sum(curr_pools[p_i][:-1])
			for j in range(N_m):
				rev[j] += float(curr_pools[p_i][j])/tot

			# clear share history of pool
			curr_pools[p_i][:-1] = N_m*[0]

		# update expected price
		prices[p_i] = 1./(D+sum(curr_pools[p_i][:-1]))

		for j in range(int(N_m*hop)):
			exp_rev = N_p*[0]
			for k in range(N_p):
				exp_rev[k] = prices[k]*(alpha[j]/stable_comp_power[k])*D
			rev_max = np.argmax(exp_rev)
			if int(rev_max) != int(loc[j]):
				# miner hops
				tot_hops += 1
				if int(loc[j]) == 0:
					loy_hops += 1
				curr_pools[int(loc[j])][N_m] -= 1
				loc[j] = rev_max
				curr_pools[int(rev_max)][N_m] += 1

	rev = rev/sum(rev)
	rev_ratio = [rev[jj]/alpha[jj] for jj in range(N_m)]
	rev_ratio_hop[ii] = sum(rev_ratio[0:int(hop*N_m)])/len(rev_ratio[0:int(hop*N_m)])
	rev_ratio_pplns[ii] = sum([rev_ratio[i] for i, x in enumerate(loc) if x == 0 and i > hop*N_m])/len([i for i, x in enumerate(loc) if x == 0 and i > hop*N_m])
	rev_ratio_fix[ii] = sum([rev_ratio[i] for i, x in enumerate(loc) if x != 0 and i > hop*N_m])/len([i for i, x in enumerate(loc) if x != 0 and i > hop*N_m])

print 'hoppers'
print sum(rev_ratio_hop)/trials
print 'pplns'
print sum(rev_ratio_pplns)/trials
print 'fix'
print sum(rev_ratio_fix)/trials

print float(loy_hops)/float(tot_hops)
# f1 = plt.figure()

# rev = rev/sum(rev)

# plt.plot(alpha[0:int(hop*N_m)], rev[0:int(hop*N_m)], 'o', label = 'hoppers')
# plt.plot(alpha[[i for i, x in enumerate(loc) if i > hop*N_m]], rev[[i for i, x in enumerate(loc) if i > hop*N_m]], 'ro', label = 'non-hoppers')
# plt.xlabel(r'$\alpha$')
# plt.ylabel('revenue')
# plt.legend(loc=2)

# plt.savefig('prop_revs.pdf')