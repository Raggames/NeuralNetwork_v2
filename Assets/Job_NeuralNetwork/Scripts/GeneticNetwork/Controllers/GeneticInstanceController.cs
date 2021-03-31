using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public abstract class GeneticInstanceController : MonoBehaviour
    {
        public int UniqueID;

        protected GeneticEvolutionManager evolutionManager;
        protected GeneticBrain geneticBrain;

        [Header("Execution Compute Rate")]
        public float Rate = 0.5f;
        protected float rateTimer;

        public void Init(GeneticBrain GeneticEntity, GeneticEvolutionManager EvolutionManager)
        {
            evolutionManager = EvolutionManager;
            geneticBrain = GeneticEntity;
            UniqueID = EvolutionManager.GetUniqueID();
        }

        public abstract void Born();

        public abstract GeneticEvaluationData ComputeEvaluationData();

        public abstract void ExecuteDecision(double[] inputs);

        public abstract void Reproduct(GeneticInstanceController partner);

        public abstract void MutateGene();

        public abstract void Die();
    }
}
