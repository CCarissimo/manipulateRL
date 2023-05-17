# manipulateRL

In this repository we explore the effects that recommender systems may have on users. This work is the code-base for a NEURIPS conference submission. We follow the formalism presented in the paper: [ANONYMIZED].

Users are modelled as Q-learning agents, with dynamic preferences, and engage in a multi-agent environment described by a Congestion game. The recommender systems are assumed to know the optimal (social welfare) solution and provide recommendations to the agents as state information (see figure below).

<img src="/plots/state_recommender.png" width="65%"/>

We take networks from the Braess Paradox to test the ideas behind the model.

<img src="/plots/braess_networks.png" width="65%"/>

## Tests in the Initial Network of the Braess Paradox

The Nash Equilibrium is aligned with the Social Welfare optimizing state, such that the average latency experienced by all agents is $1.5$.
Using the negative of latency as the rewards $(r^i=-l(a))$, we choose to initialize all agents equally with the following $Q$-values, $Q(\bullet,u)=Q(\bullet,d)= -1.5$.
These are equivalent to the negative of the latency of the actions up and down experienced by agents at the Nash Equilibrium (higher latency leads to lower $Q$-values). Due to the $\epsilon$-greedy policy, for equal $Q$-values an agent picks the first action in their $Q$-tables (the definition of the $\mathbf{argmax}$ operator being in line with NumPy). Thus, regardless of the recommendation, the $\mathbf{argmax}$ of all agents will be to go up in the first iteration of the game. The only deviations from this action will be due to the $\epsilon$ exploration rate. This creates a simple scenario for an RS to evaluate the effects of its recommendations on welfare.

<img src="/plots/recommenders_T_mean_all_alignment_all.png" width="65%"/>

**Tested cases and results for the Initial Network** 
- (**constant**) This recommendation is kept constant, and it does not help the agents to converge. In fact, keeping a recommendation constant is equivalent to providing no recommendation. Specifically it is a fixed recommendation that splits the population of agents such that half are recommended to go up, and the other half recommended to go down: $S = (u,d,u,d,\dots,u,d)$.  
- (**random**) The recommendation is randomized between each iteration. The results show great similarity with providing a constant recommendation, achieving no improvement of average latency, and poor recommendation alignment.
- (**misaligned**) A two-step recommendation, sufficient for agents to immediately converge to the Nash Equilibrium. The first recommendation is for all agents to go up. The second recommendation splits the population, as in the fixed case. The recommendation is then kept constant. While achieving rapid convergence to the Nash Equilibrium this recommendation is fully misaligned.
- (**aligned**) A two-step recommendation, as in the misaligned case, but the first recommendation is for all agents to go down. Again, the second recommendation splits the population half-and-half and is then kept constant. This recommendation achieves both rapid convergence and recommendation alignment.

## Recommenders for the Augmented Network

The next scenario we consider presents a routing network, where the Nash Equilibrium is not the Social Welfare optimizing solution. This routing network is commonly referred to as the augmented network of the Braess Paradox, first introduced from the perspective of cars, but also extended for the cases of packet routing. The Nash Equilibrium has all agents crossing, which leads to an average latency of $2$. However, the Social Welfare optimizing solution remains the same as in the previous simple scenario: half of the agents go up and the other half go down, for an average latency of $1.5$. The rewards for this game are:

### The Heuristic Recommender

The Heuristic Recommender: Agents are categorized as $up$, $down$, or $cross$ groups according to the $\mathbf{argmax}$ actions in each of their recommendation states. In the case of the $\mathbf{argmax}$ being a tie between both $up$ and $down$, these agents are classified as $up \cap down$. Recommendations are then set such that the, $up$ and $down$ categories' $Q$-values would improve most. Similarly, cross agents receive recommendations that worsen their $Q$-values most. Finally, $up \cap down$ agents are split equally between going up and down, if possible.

<img src="/plots/recommender_diagram.png" width="65%"/>

### The Aligned Heuristic Recommender

The *Aligned* Heuristic Recommender: agents are categorized as $up$, $down$, or $cross$ groups according to $\mathbf{argmax}=z_t^i$, that is that the recommendation $z_t^i$ is aligned with the $\mathbf{argmax}$ action. Agents can be classified into multiple groups. For agents that fall exclusively into a single group, they are recommended their aligned recommendation state. The agents that are both in the $up$ and $down$ groups are given (still aligned) recommendations such that they are split equally between going up and down, if possible. Agents with no aligned recommendation states are categorized as misaligned. All misaligned agents receive the same recommendation state, picked such that it is the state whose average belief change over all agents is maximized (the equation in the picture explains it best).
        
<img src="/plots/aligned_recommender_diagram.png" width="65%"/>

## Tests in the Augmented Network

In the figure below the $Q$-values of agents are initialized aligned, meaning that for all agents the argmax of the $Q$-values in a given recommendation state is the given recommendation.
<img src="/plots/recommenders_aligned_T_mean_alignment.png" width="65%"/>

In the figure below the $Q$-values of agents are initialized mis-aligned, meaning that for all agents the argmax of the $Q$-values in a given recommendation state is NOT the given recommendation.
<img src="/plots/recommenders_misaligned_T_mean_alignment.png" width="65%"/>

Lines are means of the last 2000 learning steps from a 10000 learning step run iterative run. 40 runs are averaged and the bands are 1 standard deviation above and below the mean. Different values of $\epsilon$ are tested for the $\epsilon$-greedy $Q$-learners (held constant throughout). Top row: The systems were initialized with aligned $Q$-values. The experiments show that the heuristic recommender yields the lowest latencies while using the aligned recommender results in high latencies (left column). However, the aligned recommender also maintains a high alignment with the beliefs of the agents (right column). Bottom row: Initializing misaligned $Q$-values leads to a different evolution of the system, which also allows the aligned recommender to have lower latencies than all other recommenders at $\epsilon>0.08$, while maintaining higher alignment.

**Tested cases and results for the Augmented Network**
- (**constant**) The RS produces the constant recommendation $S = (u,d,u,d,\dots,u,d)$. In practice, any constant recommendation is equivalent when all agents have their $Q$-values initialized in the same manner. Furthermore, it is identical to the case where no recommendation is provided. This case is to demonstrate that a successful RS must actively, and dynamically change recommendation states for the agents to benefit the social welfare. 
- (**random**) The RS generates a recommendation at each step that is uniformly picked at random from $S$. This case demonstrates that a random RS can have the beneficial effect of slowing down the natural learning dynamics. It also establishes a baseline recommendation alignment score. 
- (**heuristic**) The RS attempts to pick recommendations such that as many agents as possible split between up and down, and the remaining agents are recommended actions that would lead their beliefs to change favorably due to the learning dynamics. A detailed picture of this heuristic RS is described in. This RS achieves higher system welfare at the cost of recommendation alignment. The achieved performance increases the leverage of the manipulative potential of the omniscient RS system, which can be seen by poor recommendation alignment scores. The heuristic recommender results are nearly identical for average latency in the aligned (top row) and misaligned (bottom row) cases. This makes sense, as the heuristic recommender, `blind' to recommendation alignment, is not affected by the change in $Q$-value initialization of the users.
- (**aligned heuristic**) The RS attempts to pick recommendations such that as many agents as possible split between up and down, and the remaining agents are recommended actions that would lead their beliefs to change favorably due to the learning dynamics. A detailed picture of this *aligned* heuristic RS is described in. This still achieves an improvement in the system welfare while prioritizing recommendation alignment and achieving the highest recommendation alignment scores. When possible, this recommender always recommends aligned recommendations.


