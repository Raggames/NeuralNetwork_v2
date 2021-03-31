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

        public void CreateEntity()
        {
            var nn = Instantiate(instancePrefab, transform);
            var controller = nn.GetComponent<GeneticInstanceController>();
            NetworkInstances.Add(controller);
            controller.Init(nn, this);
            nn.CreateInstance();
        }

        private void StartTraining()
        {
            for (int j = 0; j < EntitiesToCreate; ++j)
            {
                NetworkInstances[j].Born();
            }
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
    }

    public struct GeneticEvaluationData
    {
        public int instanceID;
        public double[] result;
    }
}
