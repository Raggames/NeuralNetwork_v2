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
        protected GeneticEvolutionManager evolutionManager;
        protected GeneticBrain geneticBrain;

        public void Init(GeneticBrain GeneticEntity, GeneticEvolutionManager EvolutionManager)
        {
            evolutionManager = EvolutionManager;
            geneticBrain = GeneticEntity;
        }

        public abstract void StartExecution();

        public abstract GeneticEvaluationData ComputeEvaluationData();

        public abstract void ExecuteDecision(double[] inputs);

        public abstract void Reproduct(GeneticInstanceController partner);

        public abstract void Die();
    }
}
