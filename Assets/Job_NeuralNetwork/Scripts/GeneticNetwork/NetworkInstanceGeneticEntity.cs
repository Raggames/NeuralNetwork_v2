using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public class NetworkInstanceGeneticEntity : MonoBehaviour
    {
        [Header("Feed Forward Network")]
        public NeuralNetwork FFNetwork;
        public GeneticEvolutionManager GeneticEvolutionManager;
        public int UniqueID;

        private GeneticInstanceController instanceController;

        [Header("RunMode")]
        public RunningMode Mode;
        public enum RunningMode
        {
            Train,
            Execute,
        }
        [Header("Save Management")]
        public bool Save;
        public int AutoSaveEveryEpochs = 5000;

        [Header("Training Parameters")]
        public bool AutomaticStop;
        public int Epochs; // trainData lenght
        public int currentEpoch;
        public int BatchSize; // every nbr of runs would you want to compute error

        [Range(0.00001f, 0.5f)] public float LearningRate;
        public double Momentum;

        public double WeightDecay = 0.0001f;

        private WaitForSeconds delay;
        public float DelayBetweenEpochs = 0.05f;
        private Coroutine ExecutionCoroutine;

        [Header("Learning Rate Decay")]

        public float DecayRate = 0.95f;
        public int DecayStep;

        // **************************************************************************************
        public double[][] TestingDataX;
        public double[][] TestingDataY;

        [Header("Real Time In/Out")]
        public double[] runInputs;
        public double[] runResults;

        public bool IsExecuting;

        [Header("Execution Compute Rate")]
        public float Rate = 0.5f;
        private float rateTimer;

        public void CreateInstance()
        {
            delay = new WaitForSeconds(DelayBetweenEpochs);

            FFNetwork.CreateNetwork(null, this);
            UniqueID = GeneticEvolutionManager.GetUniqueID();
            instanceController = GetComponent<GeneticInstanceController>();
            instanceController.Init(this, GeneticEvolutionManager);
        }

        public void Execute()
        {
            FFNetwork.LoadAndSetWeights();
            //ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
        }

        public void StartExecuting()
        {
            IsExecuting = true;

           
        }

        public void Update()
        {
            if (IsExecuting)
            {
                rateTimer += Time.deltaTime;
                if(rateTimer > Rate)
                {
                    FFNetwork.ComputeFeedForward(runInputs, out runResults);

                    instanceController.ExecuteDecision(runResults);
                    rateTimer = 0;
                }
            }
        }
    }
}
