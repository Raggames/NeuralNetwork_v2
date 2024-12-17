using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.UtilityBasedAgents
{
    /// <summary>
    /// An utility related with the distance with each other agent in range
    /// </summary>
    [Serializable]
    public class FearUtility : IUtility
    {
        [SerializeField, LearnedParameter] private float _i1;
        [SerializeField, LearnedParameter] private float _x1;
        [SerializeField, LearnedParameter] private float _b1;
        [SerializeField, LearnedParameter] private float _b2;
        [SerializeField, LearnedParameter] private float _threshold;

        public double ComputeWeight(AIAgent agent)
        {
            var fear_ratio = 0f;

            for (int i = 0; i < agent.AgentsInRange.Count; ++i)
            {
                var dist = (agent.transform.position - agent.AgentsInRange[i].transform.position).magnitude;

                var base_ratio = dist * _x1 + _b1;

                // inversion parameter, if below 0 the function is inverted
                if (_i1 < 0)
                    fear_ratio += 1f / base_ratio;
                else
                    fear_ratio += base_ratio;
            }

            fear_ratio += _b2;

            return fear_ratio >= _threshold ? fear_ratio : 0f;
        }
    }
}
