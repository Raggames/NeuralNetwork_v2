using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.UtilityBasedAgents
{
    public class HungerUtility : IUtility
    {
        [SerializeField, LearnedParameter] private float _x1;
        [SerializeField, LearnedParameter] private float _b1;
        [SerializeField, LearnedParameter] private float _threshold;

        public double ComputeWeight(AIAgent agent)
        {
            var hunger_ratio = agent.CurrentHunger / agent.MaxHunger;

            var utility = hunger_ratio * _x1 + _b1;

            return utility > _threshold ? utility : 0;
        }
    }
}
