import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

D = 200 # difficulty of share being a block

N_p = 5 # number of pools

N_m = 200 # number of miners

T1 = 1000 # time period for initialization
T2 = 100*T1 # time period for simulatio

# assign a certain proportion of miners to be pool-hopping
hop = 0.20 # so say first hop % of miners are hoppers, rest are not

trials = 25
rev_ratio_hop = trials*[0.]
rev_ratio_pplns = trials*[0.]
rev_ratio_fix = trials*[0.]

for ii in range(trials):

	# set up PPLNS pool
	N = 2*D # parameter for pay-per-last-N-shares
	last_N = []

	# randomly assign computing power for each miner
	r = np.random.rand(N_m,1)
	alpha = r/sum(r)

	# revenue of each miner
	rev = np.zeros((N_m,1))

	# sample finder of each share
	share_winners = np.random.choice(N_m, T1+T2, p = alpha[:,0].tolist())

	# keep track of current population and share distribution of each pool
	curr_pools = [(N_m+1)*[0] for i in range(N_p)]
	# curr_pools[pool number] = [list with number of current shares owned by each miner and last number is current population] 

	# keep track of current computational power of each pool
	curr_comp_power = np.zeros((N_p,1))
	stable_comp_power = np.zeros((N_p,1))

	# keep track of population dynamics
	pops = np.zeros((T2+1,N_m))

	# expected prices per share
	prices = N_p*[0]

	# current location of each miner
	loc = np.zeros((N_m,1))

	# initialize population of pools

	# assign each miner uniformly at random to each pool
	for k in range(N_m):
		p = np.random.randint(N_p)
		curr_pools[p][N_m] += 1
		pops[0,p] += 1
		#curr_comp_power[p] += alpha[k]
		loc[k] = p
		if k > hop*N_m:
			stable_comp_power[p] += alpha[k]

	print stable_comp_power
	found_blocks = np.zeros((N_p,1))

	# run T1 steps to get into an "initial" state
	# in these steps, no one can pool hop - populations fixed
	for i in range(T1):
		m_i = int(share_winners[i]) # miner of share
		p_i = int(loc[m_i]) # current pool of miner
		# only keep track of last N for pool 0
		curr_pools[p_i][m_i] += 1 # give credit
		if p_i == 0:
			last_N.insert(0,m_i)
			if len(last_N) > N:
				curr_pools[p_i][last_N[-1]] -= 1
				last_N = last_N[0:N]

		check_block = np.random.rand()
		if check_block <= 1./D:
			# share was a valid block
			found_blocks[p_i] += 1
			# update revenues
			tot = sum(curr_pools[p_i][:-1])
			
			for j in range(N_m):
				rev[j] += float(curr_pools[p_i][j])/tot

			if p_i != 0:
				# clear share history of pool
				curr_pools[p_i][:-1] = N_m*[0]

	# compute current expected price per share of each pool
	for i in range(N_p):
		prices[i] = 1./(D+sum(curr_pools[i][:-1]))
		prices[0] = 1./N
		# will replace personalized expected price later

	# keep track of total hops versus hops from loyalty pool
	tot_hops = 0
	loy_hops = 0
	enter_hops = 0

	# run T2 steps for simulation
	# people may pool hop after every found share
	for i in range(T2):
		m_i = int(share_winners[i]) # miner of share
		p_i = int(loc[m_i]) # current pool of miner
		# only keep track of last N for pool 0
		curr_pools[p_i][m_i] += 1 # give credit
		if p_i == 0:
			last_N.insert(0,m_i)
			if len(last_N) > N:
				curr_pools[p_i][last_N[-1]] -= 1
				last_N = last_N[0:N]
		check_block = np.random.rand()
		if check_block <= 1./D:
			# share was a valid block
			found_blocks[p_i] += 1
			# update revenues
			tot = sum(curr_pools[p_i][:-1])
			for j in range(N_m):
				rev[j] += float(curr_pools[p_i][j])/tot

			# clear share history of pool
			if p_i != 0:
				curr_pools[p_i][:-1] = N_m*[0]

		# update expected price per share
		# note if p_i = 0, then updating doesn't matter
		prices[p_i] = 1./(D+sum(curr_pools[p_i][:-1]))
		prices[0] = 1./N
		# allow people to hop

		for j in range(int(N_m*hop)):
			# compute expected rev for each pool over next D shares found
			exp_rev = N_p*[0]
			for k in range(N_p):
				exp_rev[k] = prices[k]*(alpha[j]/stable_comp_power[k])*D
			rev_max = np.argmax(exp_rev)
			if int(rev_max) != int(loc[j]):
				# miner hops
				tot_hops += 1
				if int(loc[j]) == 0:
					loy_hops += 1
				if int(rev_max) == 0:
					enter_hops += 1
				curr_pools[int(loc[j])][N_m] -= 1
				loc[j] = rev_max
				curr_pools[int(rev_max)][N_m] += 1

		# track population
		for j in range(N_p):
			pops[i+1,j] = curr_pools[j][N_m]


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
print float(enter_hops)/float(tot_hops)

# f1 = plt.figure()

# rev = rev/sum(rev)

# plt.plot(alpha[0:int(hop*N_m)], rev[0:int(hop*N_m)], 'o', label = 'hoppers')
# plt.plot(alpha[[i for i, x in enumerate(loc) if x != 0 and i > hop*N_m]], rev[[i for i, x in enumerate(loc) if x != 0 and i > hop*N_m]], 'ro', label = 'non-hoppers')
# plt.plot(alpha[[i for i, x in enumerate(loc) if x == 0 and i > hop*N_m]], rev[[i for i, x in enumerate(loc) if x == 0 and i > hop*N_m]], 'go', label = 'non-hoppers, loyalty')
# plt.xlabel('alpha')
# plt.ylabel('rev')
# plt.legend()

# f2 = plt.figure()

# #for k in range(N_p):
# #	plt.plot(range(T2+1), pops[:,k]/N_m, label = '%d' % k)
# #plt.legend()

# plt.plot(range(T2+1), pops[:,0]/N_m)
# plt.show()