
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.AI;

namespace Atom.MachineLearning.MiniProjects.ReinforcementUtilityBasedAgents
{
    internal class MoveRandomAction : ActionBehaviourBase
    {
        [SerializeField] private float _speed = 4;
        
        public override List<IUtility> Utilities => new List<IUtility>()
        {
            new FearUtility(),
            new HungerUtility(),
            new LifeUtility(),
        };

        public override void Execute(AIAgent agent)
        {
           
        }
    }
}
