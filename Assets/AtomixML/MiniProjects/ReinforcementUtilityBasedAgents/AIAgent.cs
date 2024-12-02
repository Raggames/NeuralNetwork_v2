using NUnit.Framework;
using Sirenix.OdinInspector;
using System.Collections.Generic;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.ReinforcementUtilityBasedAgents
{
    public class AIAgent : MonoBehaviour
    {
        // hunger grow with time 
        // agent have to eat other agent to dimnish hunger
        [SerializeField] private float _currentHunger = 0;
        [SerializeField] private float _maxHunger = 100;
        [SerializeField] private float _hungerGrowingRatio = .1f; // .1f per second 

        public float CurrentHunger {get => _currentHunger; set => _currentHunger = value; }
        public float MaxHunger => _maxHunger;

        [SerializeField] private float _currentLife = 0;
        [SerializeField] private float _maxLife = 100;
        [SerializeField] private float _hungerDamageRatio = .1f; // 1 point of hunger = .1f damage per second

        public float CurrentLife {get => _currentLife; set => _currentLife = value; }
        public float MaxLife => _maxLife;

        [SerializeField] private float _detectionRange = 8;

        private List<AIAgent> _agentsInRange = new List<AIAgent>();
        public List<AIAgent> AgentsInRange => _agentsInRange;

        [ShowInInlineEditors, ReadOnly] private ActionBehaviourBase[] _actions;

        private void Awake()
        {
            _actions = GetComponents<ActionBehaviourBase>();
        }

        private void Update()
        {
            _currentHunger += Time.deltaTime * _hungerGrowingRatio;
            _currentLife -= Time.deltaTime * _hungerDamageRatio * _currentHunger;

            int best_rank_action = -1;
            double best_rank = float.MinValue;
            for (int i = 0; i < _actions.Length; ++i)
            {
                var crt_rank = _actions[i].Rank(this);
                if(crt_rank > best_rank)
                {
                    best_rank_action = i;
                    best_rank = crt_rank;
                }
            }

            _actions[best_rank_action].Execute(this);
        }
    }
}
