using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace NeuralNetwork
{
    public abstract class NeuralNetworkTrainer : MonoBehaviour
    {
        [Header("---- TRAINING SETTING")]
        /// <summary>
        /// Contains the data set and the function to evaluate the accuracy of the network while training
        /// </summary>
        public TrainingSettingBase TrainingSetting;
        [Header("---- SAVE ----")]
        public string SaveName = "";

        [Header("---- PARAMETERS ----")]
        /// <summary>
        /// The range of the weigths at initialization
        /// </summary>
        public Vector2 InitialWeightRange = new Vector2(-.1f, .1f);
        /// <summary>
        /// The random seed for initializing weigths
        /// </summary>
        public int InitialWeightSeed = 0;
        /// <summary>
        /// The multiplier for tweaking weights each epoch/batch
        /// </summary>
        [Range(0.00001f, 2f)] public float LearningRate = .3f;

        /// <summary>
        /// Momentum is a ratio that allows to add to the new computed weights a part of the vector of the previous iteration.
        /// It allows the learning to be faster by decreasing the fluctuations of the process.
        /// </summary>
        public float Momentum = 0.01f;

        /// <summary>
        /// The learning rate for the biases
        /// </summary>
        public float BiasRate = 1;

        /// <summary>
        /// Control the decreasing of the learning rate over epochs
        /// </summary>
        public float LearningRateDecay = 0.0001f;

        /// <summary>
        /// Decreases all the weigths of the network over epochs (should be a very small amount, it can totally cancel the learning if to close to the learning rate.
        /// </summary>
        public float WeightDecay = 0.0001f;

        /// <summary>
        /// Number of iterations of the learning process
        /// </summary>
        public int Epochs = 1;
        /// <summary>
        /// [Not yet implemented] Batch are a way to accumulate iterations before updating weights (accumulating gradients or error)
        /// </summary>
        public int BatchSize = 1;

        [Header("---- RUNTIME ----")]
        [ReadOnly] public int CurrentEpoch;
        [ReadOnly] public double Training_Best_Accuracy;

        public double[][,] Training_Best_Weigths;
        public double[][] Training_Best_Biases;

        protected void InitializeTrainingBestWeightSet(NeuralNetwork neuralNetwork)
        {
            // Each epoch, we keep a trace of the best set
            // If the next one doesn't find a best set, we will retry from this one
            // If the retry iterations are over a threshold, the learning will stop and keed this set as the best
            Training_Best_Weigths = new double[neuralNetwork.layers.Count][,];
            Training_Best_Biases = new double[neuralNetwork.layers.Count][];

            for (int i = 0; i < neuralNetwork.layers.Count; ++i)
            {
                Training_Best_Weigths[i] = NeuralNetworkMathHelper.MakeMatrix(neuralNetwork.layers[i].Weights.GetLength(0), neuralNetwork.layers[i].Weights.GetLength(1));
                Training_Best_Biases[i] = new double[neuralNetwork.layers[i].Biases.Length];
            }
        }

        protected void MemorizeBestSet(NeuralNetwork bestSet, double accuracy)
        {
            Training_Best_Accuracy = accuracy;

            for (int l = 0; l < bestSet.layers.Count; ++l)
            {
                for (int i = 0; i < bestSet.layers[l].Weights.GetLength(0); ++i)
                {
                    for (int k = 0; k < bestSet.layers[l].Weights.GetLength(1); ++k)
                    {
                        Training_Best_Weigths[l][i, k] = bestSet.layers[l].Weights[i, k];
                    }
                }

                for (int i = 0; i < bestSet.layers[l].Biases.Length; ++i)
                {
                    Training_Best_Biases[l][i] = bestSet.layers[l].Biases[i];
                }
            }
        }

        protected void SetWeightsFromBest(NeuralNetwork network)
        {
            for (int l = 0; l < network.layers.Count; ++l)
            {
                for (int i = 0; i < network.layers[l].Weights.GetLength(0); ++i)
                {
                    for (int k = 0; k < network.layers[l].Weights.GetLength(1); ++k)
                    {
                        network.layers[l].Weights[i, k] = Training_Best_Weigths[l][i, k];
                    }
                }

                for (int i = 0; i < network.layers[l].Biases.Length; ++i)
                {
                    network.layers[l].Biases[i] = Training_Best_Biases[l][i];
                }
            }
        }

        /// <summary>
        /// Pass any network as reference, its weight will be reset to the best training values and saved
        /// </summary>
        /// <param name="referent"></param>
        protected void SaveBestTrainingWeightSet(NeuralNetwork referent)
        {
            SetWeightsFromBest(referent);
            
            double[] weights = referent.GetWeights();

            string version = DateTime.Now.ToShortDateString() + referent.ArchitectureString() + "_BestSet";

            NetworkData data = new NetworkData
            {
                Version = version,
                dnaSave = weights,
                learningRate = LearningRate,

                momentum = Momentum,
                weightDecay = WeightDecay,
                accuracy = (float)Training_Best_Accuracy,
            };

            NetworkDataSerializer.Save(data, data.Version);
        }

        protected NetworkData LoadDataByName(string fileName)
        {
            NetworkData loadedData = new NetworkData();
            loadedData = NetworkDataSerializer.Load(loadedData, fileName);
            return loadedData;
        }
    }
}
