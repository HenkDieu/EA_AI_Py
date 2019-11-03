import util

#Initialize HMM

K = 26 + 1 # 1 for space

#startProbs: p_start(h)

startProbs = [1.0/K for _ in range(K)]

#transProbs: p_trans(h2 | h1]
#Estimate from raw text (fully supervised)

rawText = util.toIntSeq(util.readText('lm.train'))
transCounts = [[0 for h2 in range(k)] for h2 in range(K)]
for i in range(len(rawText)-1):
    transCounts[rawText[i]][rawText[i+1]] += 1

transProbs = map(util.normalize, transCounts)
# emissionProbs: p_emit(e|h)

emissionProbs = [[1.0/K for e in range(K)] for h in range(K)]

# Run EM

observations = util.toIntSeq(util.readText('ciphertext'))

for t in range(200):
    # E-step
    # q[i][h] = P(H_i = h | E = observations)

    q = util.forwardBackward(observations, startProbs. transProbs, emissionProbs)
    # Print out best guess
    print(util.toStrSeq(map(util.argmax, q)))

    # M-step
    emissionCounts = [[0 for e in range(K)] for h in range(K)]
    for i in range(len(observations)):
        for h in range(K):
            emissionCounts[h][observations[i]] += q[i][h]
    emissionProbs = map(util.normalize, emissionCounts)