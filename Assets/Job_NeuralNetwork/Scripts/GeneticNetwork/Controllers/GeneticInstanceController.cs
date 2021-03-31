using Assets.Job_NeuralNetwork.Scripts.GeneticNetwork;
using Assets.Job_NeuralNetwork.Scripts.GeneticNetwork.Controllers;
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
     
        [Header("References")]
        protected GeneticEvolutionManager evolutionManager;
        protected GeneticBrain geneticBrain;
        protected Memory memory;

        [Header("Traits")]
        public List<Gene> Traits = new List<Gene>();

        [Header("Evaluation Parameters")] // Some more meta-parameters to evaluate fitness to an entity in its environnement
        public float SurvivedTime;
        public float FoodEaten; // amount of currentHunger --
        public float NumberOfChilds;

        [Header("Execution Compute Rate")]
        public float Rate = 0.5f;
        protected float rateTimer;

        public virtual void Init(GeneticBrain GeneticEntity, GeneticEvolutionManager EvolutionManager, List<Gene> DnaTraits)
        {
            evolutionManager = EvolutionManager;
            geneticBrain = GeneticEntity;
            memory = GetComponent<Memory>();

            UniqueID = EvolutionManager.GetUniqueID();
        }

        public abstract void Born();

        public abstract GeneticEvaluationData ComputeEvaluationData();

        public abstract void ExecuteDecision(double[] inputs);

        public abstract bool AskReproduction(GeneticInstanceController fromPartner);

        public abstract void Reproduct();

        public abstract void MutateGene(Gene gene);

        public abstract void Die();
    }
}
