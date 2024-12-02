using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.ReinforcementUtilityBasedAgents
{
    public class WaitAction : ActionBehaviourBase
    {
        public override List<IUtility> Utilities => new List<IUtility>()
        {
            new FearUtility(),
            new HungerUtility(),
            new LifeUtility(),
        };

        public override void Execute(AIAgent agent)
        {
           // do nothing
        }
    }
}
