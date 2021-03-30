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
        protected NetworkInstanceGeneticEntity geneticEntity;
        protected GeneticEvolutionManager evolutionManager;
        public void Init(NetworkInstanceGeneticEntity GeneticEntity, GeneticEvolutionManager EvolutionManager)
        {
            geneticEntity = GeneticEntity;
            evolutionManager = EvolutionManager;
        }

        public abstract GeneticEvaluationData ComputeEvaluationData();

        public abstract void ExecuteDecision(double[] inputs);

        public abstract void Reproduct(GeneticInstanceController partner);

        public abstract void Die();
    }
}
