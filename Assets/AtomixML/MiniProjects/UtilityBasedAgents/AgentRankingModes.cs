using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.AtomixML.MiniProjects.ReinforcementUtilityBasedAgents
{
    public enum AgentRankingModes
    {
        /// <summary>
        /// The agent will always use the best rank action at any execution
        /// </summary>
        BestUtility,

        /// <summary>
        /// The agent will use a random function where the rank is a normalized probability of executing
        /// 
        /// Stochastic ranking could be though as an epsilon greedy exploration/exploitation implementation in the context of hybrid training. 
        /// It will ensure that all possibilities.
        /// </summary>
        Stochastic,

        /// <summary>
        /// Possible overrides of the rank function
        /// </summary>
        Custom,
    }
}
