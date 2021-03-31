using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public class GeneticBrain : MonoBehaviour
    {
        [Header("Feed Forward Network")]
        public NeuralNetwork FFNetwork;
        public GeneticEvolutionManager GeneticEvolutionManager;

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

      

        public void CreateInstance()
        {
            delay = new WaitForSeconds(DelayBetweenEpochs);

            FFNetwork.CreateNetwork(null, this);
        }

        public void LoadBrain()
        {
            FFNetwork.LoadAndSetWeights();
            //ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
        }

        public void SaveBrain()
        {
            // string saveName => Genetic_Animal_uniqueID_NNArchitecture
            // save the saveName in playerprefs to automatically load
            // CreateFolder for it
            FFNetwork.GetAndSaveWeights(LearningRate, Momentum, WeightDecay);
        }

        public double[] Compute(double[] inputs)
        {
            FFNetwork.ComputeFeedForward(inputs, out runResults);
            return runResults;
        }
    }
}
