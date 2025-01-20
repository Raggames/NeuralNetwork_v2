using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    [Serializable]
    public class PIDControlledDroneGeneticTrainer : GeneticOptimizerBase<SimulatedPIDControlledDrone>
    {
        [SerializeField] private SimulatedPIDControlledDrone _pf_drone;

        [Header("Initial values")]
        [SerializeField] private float _trs_P = 1.25f;
        [SerializeField] private float _trs_I = .45f;
        [SerializeField] private float _trs_D = .125f;
        [SerializeField] private int _trs_DRange;
        [SerializeField] private float _translationSensivity = .1f;
        [SerializeField] private float _rotationSensivity = .33f;


        public override async Task ComputeGeneration()
        {
            while (true)
            {
                int count_alive = 0;

                foreach (var entity in CurrentGenerationEntities)
                {
                    if (!entity.enabled)
                    {
                        count_alive++;
                    }
                }

                if (count_alive == 0)
                {
                    break;
                }

                await Task.Delay(1000);
            }

        }

        public override SimulatedPIDControlledDrone CreateEntity()
        {
            var entity = GameObject.Instantiate(_pf_drone);

            entity.Parameters = new NVector()
            {
                Data = new double[6]
                {
                    _trs_P + MLRandom.Shared.Range(-.5, .5),
                    _trs_I + MLRandom.Shared.Range(-.5, .5),
                    _trs_D + MLRandom.Shared.Range(-.5, .5),
                    _trs_DRange+ MLRandom.Shared.Range(-2, .2),
                    _translationSensivity + MLRandom.Shared.Range(-.5, .5),
                    _rotationSensivity + MLRandom.Shared.Range(-.5, .5),
                }
            };

            return entity;
        }

        public override double GetEntityScore(SimulatedPIDControlledDrone entity)
        {
            return entity.currentReward;
        }

        public override void OnObjectiveReached(SimulatedPIDControlledDrone bestEntity)
        {
            Debug.Log("Objective reached");

        }

        protected override void ClearPreviousGeneration(List<SimulatedPIDControlledDrone> previousGenerationEntities)
        {
            foreach (var entity in previousGenerationEntities)
                GameObject.Destroy(entity.gameObject);
        }
    }
}
