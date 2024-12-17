using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.UtilityBasedAgents
{
    [Serializable]
    /// <summary>
    /// Goes over avalaible other agent to compute a score depending on the context.
    /// For an aggressive action, the RankTarget could be fit to output higher score for low life agents (considered as a fastes way to gain life/points)
    /// </summary>
    public class RankAgentTargetUtility : IUtility
    {
        /// <summary>
        /// A buffer of the previous computation
        /// </summary>
        [SerializeField] private List<AIAgent> _currentRankedAgents = new List<AIAgent>();

        public List<AIAgent> currentRankedAgents => _currentRankedAgents;

        public double ComputeWeight(AIAgent agent)
        {
            throw new NotImplementedException();
        }
    }
}
