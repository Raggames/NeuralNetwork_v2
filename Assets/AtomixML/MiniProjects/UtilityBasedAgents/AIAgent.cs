using NUnit.Framework;
using Sirenix.OdinInspector;
using System.Collections.Generic;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.UtilityBasedAgents
{
    /// <summary>
    /// To do create an abstraction for Statisitcs (Life, Hunger, Energy)
    /// We want to generalise this system to more sophisticated agents later, if it works well
    /// 
    /// </summary>
    public class AIAgent : MonoBehaviour
    {
        /// <summary>
        /// Hunger will motivate the agent to execute 'eat' actions
        /// Hunger and Life are obviously related in this behaviour as going high in hunger will cause agent to loose life.
        /// </summary>
        #region Hunger

        // hunger grow with time 
        // agent have to eat other agent to dimnish hunger
        [SerializeField] private float _currentHunger = 0;
        [SerializeField] private float _maxHunger = 100;
        [SerializeField] private float _hungerGrowingRatio = .1f; // .1f per second 

        public float CurrentHunger {get => _currentHunger; set => _currentHunger = value; }
        public float MaxHunger => _maxHunger;
        #endregion

        /// <summary>
        /// Life determines the durability of the agent. At 0, the agent dies
        /// </summary>
        #region Life
        [SerializeField] private float _currentLife = 0;
        [SerializeField] private float _maxLife = 100;
        [SerializeField] private float _hungerDamageRatio = .1f; // 1 point of hunger = .1f damage per second

        public float CurrentLife {get => _currentLife; set => _currentLife = value; }
        public float MaxLife => _maxLife;
        #endregion

        /// <summary>
        /// Any action done has a cost. This energy cost is a drawback to any decision that will balance eager behaviour of juste trying to move/eat all the time
        /// </summary>
        #region Energy

        [SerializeField] private float _currentEnergy = 0;
        [SerializeField] private float _maxEnergy = 100;

        public float CurrentEnergy { get => _currentEnergy; set => _currentEnergy = value; }
        public float MaxEnergy => _maxEnergy;

        #endregion


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
