using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.ReinforcementUtilityBasedAgents
{
    /// <summary>
    /// Life utility
    /// The more life, the more utility
    /// </summary>
    [Serializable]
    public class LifeUtility : IUtility
    {
        [SerializeField, LearnedParameter] private float _x1;
        [SerializeField, LearnedParameter] private float _b1;
        [SerializeField, LearnedParameter] private float _threshold;

        public double ComputeWeight(AIAgent agent)
        {
            var life_ratio = agent.CurrentLife / agent.MaxLife;

            var utility = life_ratio * _x1 + _b1;

            return utility > _threshold ? utility : 0;
        }
    }

    /// <summary>
    /// Invert of life utility 
    /// The less current life, the more utility
    /// </summary>
    [Serializable]
    public class LifeInvertedUtility : IUtility
    {
        [SerializeField, LearnedParameter] private float _x1;
        [SerializeField, LearnedParameter] private float _b1;
        [SerializeField, LearnedParameter] private float _threshold;

        public double ComputeWeight(AIAgent agent)
        {
            var life_ratio = agent.MaxLife / (1f + agent.CurrentLife);

            var utility = life_ratio * _x1 + _b1;

            return utility > _threshold ? utility : 0;
        }
    }
}
