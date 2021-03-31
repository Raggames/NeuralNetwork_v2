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
       
        private float LearningRate;
               
        // **************************************************************************************
        public double[][] TestingDataX;
        public double[][] TestingDataY;

       [Header("Real Time In/Out")]
        public double[] runInputs;
        public double[] runResults;
              

        public void CreateInstance(GeneticEvolutionManager manager)
        {
            FFNetwork = GetComponent<NeuralNetwork>();
            GeneticEvolutionManager = manager;
            LearningRate = GeneticEvolutionManager.NeuralCrossoverRate;
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
            FFNetwork.GetAndSaveWeights(LearningRate, 0, 0);
        }

        // Weight and bias randomization and reproduction and mutations

        public double[] Compute(double[] inputs)
        {
            runInputs = inputs;
            FFNetwork.ComputeFeedForward(inputs, out runResults);
            return runResults;
        }
    }
}
