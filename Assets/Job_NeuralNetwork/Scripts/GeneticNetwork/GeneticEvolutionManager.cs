using Assets.Job_NeuralNetwork.Scripts.GeneticNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public class GeneticEvolutionManager : MonoBehaviour
    {
        private int setID = 0;
        [Header("Instances")]
        public GeneticBrain instancePrefab;

        public List<GeneticInstanceController> NetworkInstances = new List<GeneticInstanceController>();

        public int EntitiesToCreate;
        public int GenerationCount;
        public int DeadEntities;

        [Header("Entities Data Management")]
        private List<GeneticEvaluationData> InstancesData = new List<GeneticEvaluationData>();

        // ***************************************************************************************************
        private void Start()
        {
            for (int i = 0; i < EntitiesToCreate; ++i)
            {
                CreateEntity();
            }

            StartTraining();
        }

        public GeneticInstanceController CreateEntity(List<Gene> genoma = null)
        {
            var nn = Instantiate(instancePrefab, transform);
            var controller = nn.GetComponent<GeneticInstanceController>();
            NetworkInstances.Add(controller);
            controller.Init(nn, this, genoma);
            nn.CreateInstance();

            return controller;
        }

        private void StartTraining()
        {
            for (int j = 0; j < EntitiesToCreate; ++j)
            {
                NetworkInstances[j].Born();
            }
        }
        // *******************************************************************************
        #region Reproductions
                
        public void Request_ComputeReproduction(GeneticInstanceController male, GeneticInstanceController female)
        {
            List<Gene> childGenoma = new List<Gene>();

            for(int i = 0; i < male.Traits.Count; ++i) // asusming A and B have same Traits List (in order and lenght)
            {
              childGenoma.Add(CrossOver(male.Traits[i], female.Traits[i]));
            }

            var childThinkRate = CrossOver(male.ThinkRate, female.ThinkRate);
            GeneticInstanceController child = CreateEntity(childGenoma);
            child.ThinkRate = childThinkRate;

            // TODO LATER : replace by a partnerFemale.Gestate()
            child.Born();

            male.NumberOfChilds++;
            female.NumberOfChilds++;
        }

        public Gene CrossOver(Gene A, Gene B)
        {
            Gene crossedGene = new Gene();
            
            float delta = 0;

            if (Rand(A.Dominance) >= Rand(B.Dominance))
            {
                delta = A.Value - B.Value;
                crossedGene.Value = A.Value - delta * A.Dominance;
                crossedGene.Dominance = A.Dominance + 0.01f; // Adding a small value each win to dominance

                crossedGene.MutationVersion = A.MutationVersion;
                crossedGene.TraitName = A.TraitName;
            }
            else
            {
                delta = B.Value - A.Value;
                crossedGene.Value = B.Value - delta * B.Dominance;
                crossedGene.Dominance = B.Dominance + 0.01f;

                crossedGene.MutationVersion = B.MutationVersion;
                crossedGene.TraitName = B.TraitName;
            }
            return crossedGene;
        }
        #endregion
        // *******************************************************************************
        #region Utils

        public float Rand( float dominance)
        {
            float rand = UnityEngine.Random.Range(0f, 1f);
            return rand * dominance;
        }

        public int GetUniqueID()
        {
            return ++setID;
        }

        public void GetEvaluationData(GeneticEvaluationData data)
        {
            DeadEntities++;
            if(DeadEntities >= EntitiesToCreate)
            {
                GenerationCount++;
            }
            InstancesData.Add(data);

        }
        #endregion
    }

    public struct GeneticEvaluationData
    {
        public int instanceID;
        public double[] result;
    }
}
