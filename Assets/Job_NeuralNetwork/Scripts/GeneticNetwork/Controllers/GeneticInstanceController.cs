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
     
        [Header("References")] //set private later
        public GeneticEvolutionManager evolutionManager;
        public GeneticBrain geneticBrain;
        public Sense sense;

        [Header("Traits")]
        public List<Gene> Traits = new List<Gene>();
        
        [Header("Evaluation Parameters")] // Some more meta-parameters to evaluate fitness to an entity in its environnement
        public float SurvivedTime;
        public float FoodEaten; // amount of currentHunger --
        public float NumberOfChilds;

        protected float rateTimer;

        public virtual void Init(GeneticEvolutionManager EvolutionManager, List<Gene> DnaTraits, double[] neuralDna)
        {
            evolutionManager = EvolutionManager;

            sense = GetComponent<Sense>();
            geneticBrain = GetComponent<GeneticBrain>();
            geneticBrain.CreateInstance(evolutionManager);

            UniqueID = EvolutionManager.GetUniqueID();
        }

        public abstract void Born();

        public abstract GeneticEvaluationData ComputeEvaluationData();

        public abstract void GetSenseRefresh();

        public abstract void ExecuteDecision(double[] inputs);

        public abstract bool AskReproduction(GeneticInstanceController fromPartner);

        public abstract void Reproduct();

        public abstract void MutateGene(Gene gene);

        public abstract void Die();
    }
}
