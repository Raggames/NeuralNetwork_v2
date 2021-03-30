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

        private List<GeneticEvaluationData> InstancesData = new List<GeneticEvaluationData>();

        [Header("Instances")]
        public GeneticBrain instancePrefab;

        public List<GeneticInstanceController> NetworkInstances = new List<GeneticInstanceController>();

        public int InstancesToRun;
        public int TrainingEpochs;
        public int CurrentEpoch;

        private void Start()
        {
            for(int i = 0; i < InstancesToRun; ++i)
            {
                var nn = Instantiate(instancePrefab, transform);
                NetworkInstances.Add(nn.GetComponent<GeneticInstanceController>());
                nn.CreateInstance();
            }

            StartTraining();
        }

        private void StartTraining()
        {
            for(int i = 0; i < TrainingEpochs; ++i)
            {
                for(int j = 0; j < InstancesToRun; ++j)
                {
                    NetworkInstances[j].StartExecution();
                }


                CurrentEpoch++;
            }
        }

        public int GetUniqueID()
        {
            return ++setID;
        }

        public void GetEvaluationData(GeneticEvaluationData data)
        {

        }


    }

    public struct GeneticEvaluationData
    {
        public int networkInstanceID;
        public double[] result;
    }
}
