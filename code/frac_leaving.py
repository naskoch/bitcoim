import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def comp_Nb(loy_pool, shares):
	# compute number of shares owned by people with badges
	N_B = 0
	for k in loy_pool.keys():
		if loy_pool[k] == [0]:
			N_B += shares[int(k)]
	return N_B

def p_badge(loy_pool, shares, N_B, i, beta, D, alpha_m, alpha_p, loc):
	# expected revenue including future (with some-lookahead factor? - right now D)
	# and cost of switching out is built in (instead of subtracted from other price)
	# for miner i
	x = shares[i] # number of shares owned by i
	# check if i has is qualified for a badge
	if shares[-1] == 0 or (i in loy_pool.keys() and loy_pool[i] == [0]):
		b = 1 # have badge
	else:
		b = 0

	S = sum(shares[:-1])
	if S > 0:
		w = (1.+b*beta)/((S+D)*(1+beta*float(N_B)/S))
		w_til = 1./((S+D)*(1+beta*float(N_B-(b*x))/S))
	else:
		w = 1./D
		w_til = 1./D

	if loc == 0:
		p = x*(w-w_til)+(alpha_m/alpha_p)*D*w
	else:
		p = (alpha_m/alpha_p)*D*w
	return p

D = 200 # difficulty of share being a block

N_p = 5 # number of pools

N_m = 100 # number of miners

T1 = 1000 # time period for initialization
T2 = 100*T1 # time period for simulation

# assign a certain proportion of miners to be pool-hopping
hop = 0.20 # so say first hop % of miners are hoppers, rest are not

betas = np.linspace(5,5,1) # badge value
trials = 25
rev_hop = len(betas)*[0.]
rev_loy = len(betas)*[0.]
rev_fix = len(betas)*[0.]

leav_frac_avg = np.zeros((len(betas),1))
for ii in range(len(betas)):
	beta = betas[ii]
	print beta
	leav_frac = trials*[0.]
	rev_ratio_hop = trials*[0.]
	rev_ratio_loy = trials*[0.]
	rev_ratio_fix = trials*[0.]
	for jj in range(trials):
		print jj
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
		#pops = np.zeros((T2+1,N_m))

		# keep track of population of loyalty pool more in depth
		loy_pool = {}
		loy_prices = np.zeros((N_m,1))

		# expected prices per share
		prices = N_p*[0]

		# current location of each miner
		loc = np.zeros((N_m,1))

		# initialize population of pools

		# assign each miner uniformly at random to each pool
		for k in range(N_m):
			p = np.random.randint(N_p)
			curr_pools[p][N_m] += 1
			#pops[0,p] += 1
			if p == 0:
				loy_pool[k] = [0]
			loc[k] = p
			#curr_comp_power[p] += alpha[k]
			if k > hop*N_m:
				stable_comp_power[p] += alpha[k]

		#print stable_comp_power
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
			# will replace personalized expected price later

		# keep track of total hops versus hops from loyalty pool
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
				N_B = comp_Nb(loy_pool, curr_pools[0][:-1])
				if p_i == 0:
					for j in range(N_m):
						if j in loy_pool.keys() and loy_pool[j] == [0]:
							rev[j] += float(curr_pools[p_i][j])*(1.+beta)/(tot+beta*N_B)
						else:
							rev[j] += float(curr_pools[p_i][j])/(tot+beta*N_B)
				else:
					for j in range(N_m):
						rev[j] += float(curr_pools[p_i][j])/tot

				# clear share history of pool
				if p_i == 0:
					for key in loy_pool:
						loy_pool[key] = [0]
				
				curr_pools[p_i][:-1] = N_m*[0]

			# update expected price per share
			# note if p_i = 0, then updating doesn't matter
			prices[p_i] = 1./(D+sum(curr_pools[p_i][:-1]))

			# allow people to hop
			change_comp_power = np.zeros((N_p,1)) # change in computing power after people hop
			N_B = comp_Nb(loy_pool, curr_pools[0])
			for j in range(int(N_m*hop)):
				# compute expected rev for each pool over next D shares found
				exp_rev = N_p*[0]
				exp_rev[0] = p_badge(loy_pool, curr_pools[0], N_B, j, beta, D, alpha[j], stable_comp_power[0], loc[j])
				for k in range(1,N_p):
					exp_rev[k] = prices[k]*(alpha[j]/stable_comp_power[k])*D
				rev_max = np.argmax(exp_rev)
				if int(rev_max) != int(loc[j]):
					# miner hops
					tot_hops += 1
					if int(loc[j]) == 0:
						# left loyalty pool
						loy_hops += 1
						loy_pool[j].append(sum(curr_pools[0][:-1]))
					elif int(rev_max) == 0:
						# entered loyalty pool
						if j in loy_pool.keys():
							loy_pool[j].append(sum(curr_pools[0][:-1]))
						else:
							loy_pool[j] = [sum(curr_pools[0][:-1])]
					curr_pools[int(loc[j])][N_m] -= 1
					loc[j] = rev_max
					curr_pools[int(rev_max)][N_m] += 1
					#change_comp_power[int(loc[j])] -= alpha[j]
					#change_comp_power[int(rev_max)] += alpha[j]

			curr_comp_power += change_comp_power

			# track population
			#for j in range(N_p):
			#	pops[i+1,j] = curr_pools[j][N_m]

		leav_frac[jj] = float(loy_hops)/tot_hops

		rev = rev/sum(rev)
		rev_ratio = [rev[kk]/alpha[kk] for kk in range(N_m)]
		rev_ratio_hop[jj] = sum(rev_ratio[0:int(hop*N_m)])/len(rev_ratio[0:int(hop*N_m)])
		rev_ratio_loy[jj] = sum([rev_ratio[i] for i, x in enumerate(loc) if x == 0 and i > hop*N_m])/len([i for i, x in enumerate(loc) if x == 0 and i > hop*N_m])
		rev_ratio_fix[jj] = sum([rev_ratio[i] for i, x in enumerate(loc) if x != 0 and i > hop*N_m])/len([i for i, x in enumerate(loc) if x != 0 and i > hop*N_m])

	leav_frac_avg[ii] = sum(leav_frac)/trials
	rev_hop[ii] = sum(rev_ratio_hop)/trials
	rev_loy[ii] = sum(rev_ratio_loy)/trials
	rev_fix[ii] = sum(rev_ratio_fix)/trials


print 'hoppers'
print rev_hop[0]
print 'pplns'
print rev_loy[0]
print 'fix'
print rev_fix[0]

print leav_frac_avg[0]
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# f1 = plt.figure()

# plt.plot(betas, leav_frac_avg)
# plt.xlabel(r'$\beta$')
# plt.ylabel('hops from loyalty pool')

# plt.savefig('test_2.pdf')

# f2 = plt.figure()

# plt.plot(betas, rev_hop, label='hoppers')
# plt.plot(betas, rev_fix, 'r', label='non-hoppers')
# plt.plot(betas, rev_loy, label='non-hoppers, loyalty pool')
# plt.legend()
# plt.xlabel(r'$\beta$')
# plt.ylabel(r'$\frac{revenue}{\alpha}$')

# plt.savefig('test_3.pdf')