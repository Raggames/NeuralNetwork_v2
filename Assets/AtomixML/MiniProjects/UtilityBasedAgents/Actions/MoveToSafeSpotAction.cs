
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.AI;

namespace Atom.MachineLearning.MiniProjects.ReinforcementUtilityBasedAgents
{
    /// <summary>
    /// An action where the agent select a safe spot, computed as a position as far from any agent as possible (using random sampling)
    /// </summary>
    public class MoveToSafeSpotAction : ActionBehaviourBase
    {
        [SerializeField] private float _speed = 4;
        
        public override List<IUtility> Utilities => new List<IUtility>()
        {
            new FearUtility(),
            new HungerUtility(),
            new LifeUtility(),
        };

        public override async Task Execute(AIAgent agent)
        {
           
        }
    }
}
