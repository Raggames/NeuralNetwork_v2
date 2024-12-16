using Atom.MachineLearning.MiniProjects.UtilityBasedAgents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.UtilityBasedAgents
{
    public abstract class ActionBehaviourBase : MonoBehaviour, IAction
    {
        public abstract List<IUtility> Utilities { get; }

        public abstract Task Execute(AIAgent agent);

        public double Rank(AIAgent agent)
        {
            var result = 0.0;
            for(int i = 0; i < Utilities.Count; i++)
            {
                result += Utilities[i].ComputeWeight(agent);
            }

            return result;
        }
    }
}
