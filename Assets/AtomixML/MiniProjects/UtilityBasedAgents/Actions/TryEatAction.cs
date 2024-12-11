using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.ReinforcementUtilityBasedAgents
{
    public class TryEatAction : ActionBehaviourBase
    {
        [SerializeField] private float _damage = 34f;
        [SerializeField] private float _hungerPerKill = -50f;

        public override List<IUtility> Utilities => new List<IUtility>()
        {
            new FearUtility(),
            new HungerUtility(),
            new LifeUtility(),
        };

        public override async Task Execute(AIAgent agent)
        {
            var closest = agent.AgentsInRange.OrderByDescending(t => (t.transform.position - agent.transform.position).magnitude).FirstOrDefault();
            if(closest == null)
            {
                // PENALTY
                return;
            }

            closest.CurrentLife -= _damage;

            if(closest.CurrentLife <= 0)
            {
                agent.CurrentHunger += _hungerPerKill;
                closest.gameObject.SetActive(false);
            }
        }
    }
}
